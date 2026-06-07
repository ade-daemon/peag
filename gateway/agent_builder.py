"""
agent_builder.py
────────────────
Turns a plain-English description into a live AgentConfig CRD.

Flow:
  1. User types: "Search the web every morning and post a summary to Slack"
  2. We send that to the local LLM and ask it to return a JSON AgentConfig spec
  3. We validate the spec against known fields / allowed values
  4. We apply it to Kubernetes as an AgentConfig CRD
  5. We return the created agent's metadata to the caller

The LLM used is whichever model the "smart" ModelConfig points at — it needs
to follow instructions reliably.  If smart isn't warm yet PEAG spins it up
the normal way.
"""

import json
import logging
import re
import httpx
from datetime import datetime, timezone
from kubernetes import client, config as kube_config
from kubernetes.client.rest import ApiException

from state import get_state, ModelState
from scheduler import spin_up, wait_until_warm, _get_model_config

logger = logging.getLogger(__name__)

NAMESPACE = "ai"
OLLAMA_PORT = 11434

# ── Allowed values from the CRD ───────────────────────────────────────────────

ALLOWED_TOOLS = {
    "web_search", "file_reader", "code_runner",
    "rag", "slack", "email", "google_docs", "notion",
}
ALLOWED_MODELS = {"auto", "fast", "smart"}
ALLOWED_OUTPUT_TYPES = {"slack", "email", "webhook", "none"}

# ── System prompt sent to the LLM ─────────────────────────────────────────────

BUILDER_SYSTEM_PROMPT = """You are an AI agent configuration assistant for the PEAG platform.
Your job is to read a user's description of what they want an agent to do, and return a
JSON object that represents the agent's configuration.

Return ONLY valid JSON — no markdown, no backticks, no explanation. Just the JSON object.

The JSON must follow this exact structure:
{
  "name": "short human-readable name for the agent",
  "slug": "lowercase-hyphenated-identifier (used as the Kubernetes resource name, max 40 chars, no spaces)",
  "model": "auto | fast | smart",
  "systemPrompt": "detailed system prompt for the agent",
  "tools": ["web_search", "file_reader", "code_runner", "slack", "email", "google_docs", "notion", "rag"],
  "schedule": "cron expression if the user wants scheduled runs, otherwise omit",
  "scheduledPrompt": "the prompt to run on schedule, if schedule is set",
  "outputChannel": {
    "type": "slack | email | webhook | none",
    "target": "channel name, email address, or webhook URL"
  },
  "memory": {
    "enabled": true,
    "maxMessages": 20
  }
}

Rules:
- Only include tools from: web_search, file_reader, code_runner, slack, email, google_docs, notion, rag
- model must be one of: auto, fast, smart
  - Use "fast" for simple retrieval or summary tasks
  - Use "smart" for reasoning, code, or multi-step tasks
  - Use "auto" when unclear
- If the user mentions Slack, include "slack" in tools AND set outputChannel.type to "slack"
- If the user mentions email, include "email" in tools AND set outputChannel.type to "email"
- If the user mentions "every morning" or similar, set schedule to "0 8 * * 1-5" (weekdays 8am)
- If the user mentions "every day", set schedule to "0 8 * * *"
- If the user mentions "every hour", set schedule to "0 * * * *"
- Make systemPrompt detailed and specific to what the user described
- slug must be a valid Kubernetes name: lowercase letters, numbers, hyphens only, max 40 chars
- Do not include fields that are not needed (e.g. omit schedule if no recurring task is mentioned)
"""


# ── LLM call ──────────────────────────────────────────────────────────────────

async def _call_llm(description: str) -> str:
    """
    Call the smart model to parse the description into a JSON AgentConfig spec.
    Ensures the model is warm before calling.
    """
    config_name = "fast"

    # Warm the model if needed
    state = get_state(config_name)
    if state == ModelState.COLD:
        triggered = await spin_up(config_name)
        if not triggered:
            raise RuntimeError("Could not spin up the smart model for agent building")
        warm = await wait_until_warm(config_name, timeout=300)
        if not warm:
            raise RuntimeError("Smart model timed out during warm-up")
    elif state == ModelState.WARMING:
        warm = await wait_until_warm(config_name)
        if not warm:
            raise RuntimeError("Smart model did not become ready in time")

    mc_spec = _get_model_config(config_name)
    actual_model = mc_spec.get("modelName", config_name) if mc_spec else config_name
    model_url = f"http://peag-{config_name}.ai.svc.cluster.local:{OLLAMA_PORT}"

    # Keep model alive during the LLM call
    from state import update_last_request
    update_last_request(config_name)

    payload = {
        "model": actual_model,
        "messages": [
            {"role": "system", "content": BUILDER_SYSTEM_PROMPT},
            {"role": "user", "content": description},
        ],
        "stream": False,
        "temperature": 0.1,  # low temperature for deterministic structured output
    }

    async with httpx.AsyncClient(timeout=60.0) as http:
        resp = await http.post(f"{model_url}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ── JSON extraction ───────────────────────────────────────────────────────────

def _extract_json(raw: str) -> dict:
    """
    Pull JSON out of the LLM response.
    Models sometimes wrap JSON in markdown fences even when told not to.
    """
    # Strip markdown fences if present
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()

    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try finding a JSON object anywhere in the response
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return json.loads(match.group())

    raise ValueError(f"Could not extract JSON from LLM response: {raw[:300]}")


# ── Spec validation ───────────────────────────────────────────────────────────

def _validate_and_sanitize(spec: dict) -> tuple[str, dict]:
    """
    Validate LLM output against CRD constraints.
    Returns (slug, sanitized_spec).
    Raises ValueError if required fields are missing or invalid.
    """
    # Required fields
    if not spec.get("name"):
        raise ValueError("Agent spec missing 'name'")
    if not spec.get("slug"):
        raise ValueError("Agent spec missing 'slug'")
    if not spec.get("systemPrompt"):
        raise ValueError("Agent spec missing 'systemPrompt'")

    # Sanitize slug: lowercase, hyphens only, max 40 chars
    slug = re.sub(r"[^a-z0-9-]", "-", spec["slug"].lower())
    slug = re.sub(r"-+", "-", slug).strip("-")[:40]

    # Validate model
    model = spec.get("model", "auto")
    if model not in ALLOWED_MODELS:
        model = "auto"

    # Validate tools — drop unknown ones silently
    raw_tools = spec.get("tools", [])
    tools = [t for t in raw_tools if t in ALLOWED_TOOLS]

    # Validate outputChannel
    output_channel = spec.get("outputChannel", {"type": "none", "target": ""})
    if output_channel.get("type") not in ALLOWED_OUTPUT_TYPES:
        output_channel["type"] = "none"

    # Build clean spec
    clean = {
        "name": spec["name"][:80],
        "model": model,
        "systemPrompt": spec["systemPrompt"],
        "tools": tools,
        "outputChannel": output_channel,
        "memory": spec.get("memory", {"enabled": True, "maxMessages": 20}),
    }

    # Optional fields — only include if present and non-empty
    if spec.get("schedule"):
        clean["schedule"] = spec["schedule"]
    if spec.get("scheduledPrompt"):
        clean["scheduledPrompt"] = spec["scheduledPrompt"]
    if spec.get("ragSources"):
        clean["ragSources"] = spec["ragSources"]

    return slug, clean


# ── Kubernetes apply ──────────────────────────────────────────────────────────

def _apply_agent_config(slug: str, spec: dict) -> dict:
    """
    Create or update an AgentConfig CRD in Kubernetes.
    Returns the full CRD object that was created/patched.
    """
    try:
        kube_config.load_incluster_config()
    except Exception:
        kube_config.load_kube_config()

    custom = client.CustomObjectsApi()

    body = {
        "apiVersion": "peag.io/v1alpha1",
        "kind": "AgentConfig",
        "metadata": {
            "name": slug,
            "namespace": NAMESPACE,
            "labels": {
                "app": "peag",
                "managed-by": "agent-builder",
                "created-at": datetime.now(timezone.utc).strftime("%Y%m%d"),
            },
            "annotations": {
                "peag.io/created-by": "agent-builder",
                "peag.io/created-at": datetime.now(timezone.utc).isoformat(),
            },
        },
        "spec": spec,
    }

    try:
        # Try create first
        result = custom.create_namespaced_custom_object(
            group="peag.io",
            version="v1alpha1",
            namespace=NAMESPACE,
            plural="agentconfigs",
            body=body,
        )
        logger.info(f"AgentConfig '{slug}' created")
        return result
    except ApiException as e:
        if e.status == 409:
            # Already exists — patch it
            result = custom.patch_namespaced_custom_object(
                group="peag.io",
                version="v1alpha1",
                namespace=NAMESPACE,
                plural="agentconfigs",
                name=slug,
                body=body,
            )
            logger.info(f"AgentConfig '{slug}' updated (already existed)")
            return result
        raise


# ── Public entry point ────────────────────────────────────────────────────────

async def build_agent_from_description(description: str) -> dict:
    """
    Full pipeline: description → LLM → validated spec → Kubernetes CRD.

    Returns a summary dict with the created agent's details.
    Raises RuntimeError or ValueError on failure.
    """
    if not description or len(description.strip()) < 10:
        raise ValueError("Description is too short. Please describe what you want the agent to do.")

    logger.info(f"Building agent from description: {description[:100]}...")

    # Step 1: Ask the LLM to parse the description
    raw_llm_output = await _call_llm(description)
    logger.debug(f"LLM output: {raw_llm_output[:500]}")

    # Step 2: Extract and validate JSON
    try:
        spec_dict = _extract_json(raw_llm_output)
        slug, clean_spec = _validate_and_sanitize(spec_dict)
    except Exception:
        import re
        slug = re.sub(r'[^a-z0-9-]', '-', description[:30].lower()).strip('-')
        slug = re.sub(r'-+', '-', slug)
        tools = []
        desc_lower = description.lower()
        if any(w in desc_lower for w in ["search", "web", "news", "internet"]): tools.append("web_search")
        if any(w in desc_lower for w in ["file", "pdf", "csv", "document"]): tools.append("file_reader")
        if any(w in desc_lower for w in ["code", "python", "script", "run"]): tools.append("code_runner")
        if "slack" in desc_lower: tools.append("slack")
        if "email" in desc_lower: tools.append("email")
        clean_spec = {
            "name": description[:50],
            "model": "fast",
            "systemPrompt": f"You are a helpful AI assistant. Your task: {description}. Be concise and accurate.",
            "tools": tools,
            "outputChannel": {"type": "none", "target": ""},
            "memory": {"enabled": True, "maxMessages": 20},
        }
        logger.warning(f"Using fallback config for: {description[:50]}")

    # Step 3: Apply to Kubernetes
    crd_result = _apply_agent_config(slug, clean_spec)

    logger.info(f"Agent '{slug}' is live")

    return {
        "id": slug,
        "name": clean_spec["name"],
        "model": clean_spec["model"],
        "tools": clean_spec["tools"],
        "schedule": clean_spec.get("schedule"),
        "outputChannel": clean_spec.get("outputChannel"),
        "systemPrompt": clean_spec["systemPrompt"],
        "created": crd_result.get("metadata", {}).get("creationTimestamp"),
        "status": "created",
    }
