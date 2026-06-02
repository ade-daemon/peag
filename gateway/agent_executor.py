import httpx
import json
import logging
from datetime import datetime
from kubernetes import client, config as kube_config
from kubernetes.client.rest import ApiException
from state import get_state, ModelState
from scheduler import spin_up, wait_until_warm
from task_intelligence import route

logger = logging.getLogger(__name__)

NAMESPACE = "ai"
OLLAMA_PORT = 11434


def _load_kube_config():
    try:
        kube_config.load_incluster_config()
    except Exception:
        kube_config.load_kube_config()


def _get_agent_config(agent_name: str) -> dict | None:
    """Fetch an AgentConfig CRD from Kubernetes."""
    try:
        _load_kube_config()
        custom = client.CustomObjectsApi()
        obj = custom.get_namespaced_custom_object(
            group="peag.io",
            version="v1alpha1",
            namespace=NAMESPACE,
            plural="agentconfigs",
            name=agent_name,
        )
        return obj.get("spec", {})
    except ApiException as e:
        if e.status == 404:
            logger.warning(f"No AgentConfig found for {agent_name}")
        else:
            logger.error(f"Error reading AgentConfig for {agent_name}: {e}")
        return None


def _get_all_agent_configs() -> list[dict]:
    """List all AgentConfig CRDs."""
    try:
        _load_kube_config()
        custom = client.CustomObjectsApi()
        result = custom.list_namespaced_custom_object(
            group="peag.io",
            version="v1alpha1",
            namespace=NAMESPACE,
            plural="agentconfigs",
        )
        return result.get("items", [])
    except ApiException as e:
        logger.error(f"Failed to list AgentConfigs: {e}")
        return []


# ── Tool execution ────────────────────────────────────────────────────────────

async def _run_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool and return the result as a string."""
    logger.info(f"Running tool: {tool_name} with input: {tool_input}")

    match tool_name:
        case "web_search":
            return await _tool_web_search(tool_input.get("query", ""))
        case "file_reader":
            return await _tool_file_reader(tool_input.get("path", ""))
        case "code_runner":
            return await _tool_code_runner(tool_input.get("code", ""))
        case _:
            return f"Tool '{tool_name}' is not yet implemented."


async def _tool_web_search(query: str) -> str:
    """Search the web using DuckDuckGo — no API key required."""
    if not query:
        return "No search query provided."
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append(f"Title: {r['title']}\nURL: {r['href']}\nSummary: {r['body']}\n")
        if not results:
            return "No results found."
        return "\n---\n".join(results)
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return f"Web search failed: {e}"


async def _tool_file_reader(path: str) -> str:
    """Read a file from the filesystem."""
    try:
        with open(path, "r") as f:
            content = f.read(5000)  # limit to 5000 chars
        return content
    except Exception as e:
        return f"File read failed: {e}"


async def _tool_code_runner(code: str) -> str:
    """Run Python code in a subprocess sandbox."""
    import asyncio
    try:
        proc = await asyncio.create_subprocess_exec(
            "python3", "-c", code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        if stdout:
            return stdout.decode()
        if stderr:
            return f"Error: {stderr.decode()}"
        return "Code ran with no output."
    except asyncio.TimeoutError:
        return "Code execution timed out (10s limit)."
    except Exception as e:
        return f"Code execution failed: {e}"


# ── Agent loop ────────────────────────────────────────────────────────────────

async def run_agent(
    agent_name: str,
    user_message: str,
    conversation_history: list[dict] | None = None,
) -> dict:
    """
    Run an agent and return the result.

    The agent loop:
    1. Load AgentConfig
    2. Pick model (auto/fast/smart)
    3. Ensure model is warm
    4. Build messages with system prompt + history + user message
    5. Call model
    6. If model calls a tool, execute it and feed result back
    7. Repeat until model gives a final text response
    8. Return result
    """
    spec = _get_agent_config(agent_name)
    if not spec:
        return {"error": f"Agent '{agent_name}' not found"}

    system_prompt = spec.get("systemPrompt", "You are a helpful assistant.")
    tools = spec.get("tools", [])
    model_preference = spec.get("model", "auto")
    agent_display_name = spec.get("name", agent_name)

    # ── Pick model ────────────────────────────────────────────────────────────
    if model_preference == "auto":
        task_type, config_name, complexity = route(user_message, is_agent=True)
    elif model_preference == "fast":
        config_name = "fast"
    else:
        config_name = "smart"

    logger.info(f"Agent '{agent_name}' using model config: {config_name}")

    # ── Ensure model is warm ──────────────────────────────────────────────────
    state = get_state(config_name)
    if state == ModelState.COLD:
        # Clean up any stale deployment before spinning up fresh
        from scheduler import scale_to_zero
        await scale_to_zero(config_name)
        import asyncio
        await asyncio.sleep(2)
        triggered = await spin_up(config_name)
        if not triggered:
            return {"error": f"Could not spin up model '{config_name}'"}
        warm = await wait_until_warm(config_name, timeout=300)
        if not warm:
            return {"error": f"Model '{config_name}' timed out during warm-up"}

    # ── Get actual model name from config ─────────────────────────────────────
    from scheduler import _get_model_config
    mc_spec = _get_model_config(config_name)
    actual_model = mc_spec.get("modelName", config_name) if mc_spec else config_name
    model_url = f"http://peag-{config_name}.ai.svc.cluster.local:{OLLAMA_PORT}"

    # ── Build tool definitions for the model ──────────────────────────────────
    tool_definitions = []
    if "web_search" in tools:
        tool_definitions.append({
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        })
    if "code_runner" in tools:
        tool_definitions.append({
            "type": "function",
            "function": {
                "name": "code_runner",
                "description": "Execute Python code and return the output",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to run"}
                    },
                    "required": ["code"]
                }
            }
        })

    # ── Build messages ────────────────────────────────────────────────────────
    messages = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        messages.extend(conversation_history[-10:])  # last 10 messages
    messages.append({"role": "user", "content": user_message})

    # ── Agent loop (max 5 iterations to prevent infinite loops) ───────────────
    max_iterations = 5
    iteration = 0

    async with httpx.AsyncClient(timeout=120.0) as http:
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Agent loop iteration {iteration}")

            # Only add tools if using a capable model (smart)
            # tinyllama and phi3 don't support tool calling
            supports_tools = config_name == "smart" and tool_definitions
            payload = {
                "model": actual_model,
                "messages": messages,
                "stream": False,
            }
            if supports_tools:
                payload["tools"] = tool_definitions

            resp = await http.post(
                f"{model_url}/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            message = choice["message"]

            # ── Tool call? Execute and loop back ──────────────────────────────
            if choice.get("finish_reason") == "tool_calls" and message.get("tool_calls"):
                messages.append(message)
                for tool_call in message["tool_calls"]:
                    tool_name = tool_call["function"]["name"]
                    try:
                        tool_input = json.loads(tool_call["function"]["arguments"])
                    except Exception:
                        tool_input = {}
                    tool_result = await _run_tool(tool_name, tool_input)
                    logger.info(f"Tool {tool_name} result: {tool_result[:100]}...")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": tool_result,
                    })
                continue

            # ── Final response ─────────────────────────────────────────────────
            final_content = message.get("content", "")
            logger.info(f"Agent '{agent_name}' completed in {iteration} iteration(s)")

            return {
                "agent": agent_display_name,
                "model_used": config_name,
                "actual_model": actual_model,
                "iterations": iteration,
                "tools_available": tools,
                "response": final_content,
                "timestamp": datetime.utcnow().isoformat(),
            }

    return {"error": "Agent loop exceeded maximum iterations"}
