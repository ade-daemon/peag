"""
tool_registry.py
────────────────
Manages tool-to-model assignments.

What this solves:
  The Agent Studio UI lets users pick a tool (e.g. "code_runner") and say
  "always use deepseek-coder for this".  We store that mapping in Redis
  so the gateway can respect it at inference time.

  It also lets you register new tool definitions — so if you add a
  "ClaudeCode-style" tool called "advanced_code_runner", you can map it
  to whatever ModelConfig makes sense (e.g. "smart" or a dedicated
  deepseek-coder config).

Redis keys used:
  tool:assignment:{tool_name}  →  config_name   (e.g. "code_runner" → "smart")
  tool:definition:{tool_name}  →  JSON blob     (name, description, parameters)
  tool:assignments             →  JSON list of all current assignments

Built-in tools are seeded on first call to list_tools() if Redis has no data.
"""

import json
import logging
from typing import Optional

import redis

logger = logging.getLogger(__name__)

# ── Redis connection ──────────────────────────────────────────────────────────

redis_client = redis.Redis(
    host="redis.ai.svc.cluster.local",
    port=6379,
    decode_responses=True,
)

# ── Built-in tool catalogue ───────────────────────────────────────────────────
# These are always available. Custom tools can be added via the API.

BUILTIN_TOOLS = [
    {
        "name": "web_search",
        "label": "Web Search",
        "description": "Search the web using DuckDuckGo. No API key required.",
        "category": "research",
        "default_model": "fast",
        "parameters": {
            "query": {"type": "string", "description": "Search query", "required": True}
        },
    },
    {
        "name": "file_reader",
        "label": "File Reader",
        "description": "Read files from disk: txt, md, json, yaml, csv, pdf.",
        "category": "data",
        "default_model": "fast",
        "parameters": {
            "path": {"type": "string", "description": "Absolute path to the file", "required": True}
        },
    },
    {
        "name": "code_runner",
        "label": "Code Runner",
        "description": "Execute Python code in a sandboxed subprocess (10s timeout).",
        "category": "code",
        "default_model": "smart",
        "parameters": {
            "code": {"type": "string", "description": "Python code to execute", "required": True}
        },
    },
    {
        "name": "rag",
        "label": "RAG / Document Search",
        "description": "Retrieve relevant passages from a document collection.",
        "category": "research",
        "default_model": "smart",
        "parameters": {
            "query": {"type": "string", "description": "Search query", "required": True},
            "sources": {"type": "array", "description": "List of document paths or URLs", "required": False},
        },
    },
    {
        "name": "slack",
        "label": "Slack",
        "description": "Post messages to a Slack channel via webhook.",
        "category": "communication",
        "default_model": "fast",
        "parameters": {
            "channel": {"type": "string", "description": "Slack channel name", "required": True},
            "message": {"type": "string", "description": "Message to send", "required": True},
        },
    },
    {
        "name": "email",
        "label": "Email",
        "description": "Send emails via SMTP.",
        "category": "communication",
        "default_model": "fast",
        "parameters": {
            "to": {"type": "string", "description": "Recipient email address", "required": True},
            "subject": {"type": "string", "description": "Email subject", "required": True},
            "body": {"type": "string", "description": "Email body", "required": True},
        },
    },
    {
        "name": "google_docs",
        "label": "Google Docs",
        "description": "Read from or write to Google Docs (requires service account).",
        "category": "data",
        "default_model": "smart",
        "parameters": {
            "doc_id": {"type": "string", "description": "Google Doc ID", "required": True},
            "action": {"type": "string", "description": "read or append", "required": True},
            "content": {"type": "string", "description": "Content to append (if action=append)", "required": False},
        },
    },
    {
        "name": "notion",
        "label": "Notion",
        "description": "Read or write Notion pages (requires Notion API key).",
        "category": "data",
        "default_model": "smart",
        "parameters": {
            "page_id": {"type": "string", "description": "Notion page ID", "required": True},
            "action": {"type": "string", "description": "read or append", "required": True},
            "content": {"type": "string", "description": "Content to append (if action=append)", "required": False},
        },
    },
]

# ── Seed built-in tools if Redis is empty ────────────────────────────────────

def _seed_builtins_if_needed() -> None:
    """Write built-in tool definitions to Redis if they haven't been written yet."""
    for tool in BUILTIN_TOOLS:
        key = f"tool:definition:{tool['name']}"
        try:
            if not redis_client.exists(key):
                redis_client.set(key, json.dumps(tool))
                logger.debug(f"Seeded built-in tool: {tool['name']}")
        except Exception as e:
            logger.warning(f"Could not seed tool {tool['name']}: {e}")


# ── Public API ────────────────────────────────────────────────────────────────

def list_tools() -> list[dict]:
    """
    Return all tool definitions (built-in + custom) with their current
    model assignments merged in.
    """
    _seed_builtins_if_needed()

    tools = []
    try:
        keys = redis_client.keys("tool:definition:*")
        for key in sorted(keys):
            raw = redis_client.get(key)
            if not raw:
                continue
            tool = json.loads(raw)
            # Merge in the current assignment (may differ from default_model)
            assignment = redis_client.get(f"tool:assignment:{tool['name']}")
            tool["assigned_model"] = assignment or tool.get("default_model", "auto")
            tools.append(tool)
    except Exception as e:
        logger.error(f"list_tools error: {e}")
        # Fall back to in-memory built-ins
        for tool in BUILTIN_TOOLS:
            t = dict(tool)
            t["assigned_model"] = t.get("default_model", "auto")
            tools.append(t)

    return tools


def get_tool(tool_name: str) -> Optional[dict]:
    """Return a single tool definition with its current assignment, or None."""
    _seed_builtins_if_needed()
    try:
        raw = redis_client.get(f"tool:definition:{tool_name}")
        if not raw:
            return None
        tool = json.loads(raw)
        assignment = redis_client.get(f"tool:assignment:{tool_name}")
        tool["assigned_model"] = assignment or tool.get("default_model", "auto")
        return tool
    except Exception as e:
        logger.error(f"get_tool error for {tool_name}: {e}")
        return None


def assign_tool_to_model(tool_name: str, model_config_name: str) -> dict:
    """
    Assign a tool to a model config.
    e.g. assign_tool_to_model("code_runner", "smart")

    Returns the updated tool definition.
    Raises ValueError if the tool doesn't exist.
    """
    tool = get_tool(tool_name)
    if tool is None:
        raise ValueError(f"Tool '{tool_name}' not found")

    try:
        redis_client.set(f"tool:assignment:{tool_name}", model_config_name)
        logger.info(f"Tool '{tool_name}' assigned to model '{model_config_name}'")
    except Exception as e:
        logger.error(f"assign_tool_to_model error: {e}")
        raise RuntimeError(f"Failed to save assignment: {e}")

    tool["assigned_model"] = model_config_name
    return tool


def assign_tools_bulk(assignments: list[dict]) -> list[dict]:
    """
    Assign multiple tools at once.
    assignments = [{"tool": "code_runner", "model": "smart"}, ...]

    Returns a list of updated tool definitions.
    Raises ValueError if any tool doesn't exist.
    """
    # Validate all first — don't partially apply
    for a in assignments:
        tool_name = a.get("tool")
        if not tool_name:
            raise ValueError("Each assignment must have a 'tool' field")
        if get_tool(tool_name) is None:
            raise ValueError(f"Tool '{tool_name}' not found")
        if not a.get("model"):
            raise ValueError(f"Assignment for '{tool_name}' is missing 'model' field")

    # Apply all
    results = []
    for a in assignments:
        updated = assign_tool_to_model(a["tool"], a["model"])
        results.append(updated)

    logger.info(f"Bulk assignment applied: {len(results)} tool(s) updated")
    return results


def register_custom_tool(
    name: str,
    label: str,
    description: str,
    category: str,
    default_model: str,
    parameters: dict,
) -> dict:
    """
    Register a new custom tool in the registry.
    This allows the Agent Studio to expose new tool types.

    name must be a valid identifier: lowercase letters, numbers, underscores.
    Raises ValueError on invalid input.
    """
    import re
    if not re.match(r"^[a-z][a-z0-9_]{1,49}$", name):
        raise ValueError(
            "Tool name must be lowercase letters, numbers, or underscores, "
            "2-50 chars, starting with a letter."
        )

    if name in {t["name"] for t in BUILTIN_TOOLS}:
        raise ValueError(f"'{name}' is a built-in tool and cannot be overwritten.")

    tool = {
        "name": name,
        "label": label,
        "description": description,
        "category": category,
        "default_model": default_model,
        "parameters": parameters,
        "custom": True,
    }

    try:
        redis_client.set(f"tool:definition:{name}", json.dumps(tool))
        logger.info(f"Custom tool '{name}' registered")
    except Exception as e:
        logger.error(f"register_custom_tool error: {e}")
        raise RuntimeError(f"Failed to register tool: {e}")

    tool["assigned_model"] = default_model
    return tool


def delete_custom_tool(tool_name: str) -> None:
    """
    Delete a custom tool from the registry.
    Cannot delete built-in tools.
    Raises ValueError if tool not found or is built-in.
    """
    tool = get_tool(tool_name)
    if tool is None:
        raise ValueError(f"Tool '{tool_name}' not found")
    if not tool.get("custom"):
        raise ValueError(f"'{tool_name}' is a built-in tool and cannot be deleted")

    try:
        redis_client.delete(f"tool:definition:{tool_name}")
        redis_client.delete(f"tool:assignment:{tool_name}")
        logger.info(f"Custom tool '{tool_name}' deleted")
    except Exception as e:
        logger.error(f"delete_custom_tool error: {e}")
        raise RuntimeError(f"Failed to delete tool: {e}")


def get_model_for_tool(tool_name: str) -> str:
    """
    Get the currently assigned model config name for a tool.
    Falls back to the tool's default_model, then to "auto".
    Used by the agent executor to respect manual assignments.
    """
    try:
        assignment = redis_client.get(f"tool:assignment:{tool_name}")
        if assignment:
            return assignment
    except Exception as e:
        logger.warning(f"Redis error in get_model_for_tool: {e}")

    # Fall back to built-in default
    for tool in BUILTIN_TOOLS:
        if tool["name"] == tool_name:
            return tool["default_model"]

    return "auto"
