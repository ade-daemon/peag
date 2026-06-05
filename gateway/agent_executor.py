import httpx
import json
import logging
from datetime import datetime, timezone
from kubernetes import client, config as kube_config
from kubernetes.client.rest import ApiException
from state import get_state, ModelState, update_last_request
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


# ── Tool execution ────────────────────────────────────────────────────────────────────────────

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
        case "rag":
            return await _tool_rag(
                tool_input.get("query", ""),
                tool_input.get("sources", []),
            )
        case "slack":
            return await _tool_slack(
                tool_input.get("channel", ""),
                tool_input.get("message", ""),
            )
        case "email":
            return await _tool_email(
                tool_input.get("to", ""),
                tool_input.get("subject", ""),
                tool_input.get("body", ""),
            )
        case "google_docs":
            return await _tool_google_docs(
                tool_input.get("doc_id", ""),
                tool_input.get("action", "read"),
                tool_input.get("content", ""),
            )
        case "notion":
            return await _tool_notion(
                tool_input.get("page_id", ""),
                tool_input.get("action", "read"),
                tool_input.get("content", ""),
            )
        case _:
            return f"Tool '{tool_name}' is not recognised. Available tools: web_search, file_reader, code_runner, rag, slack, email, google_docs, notion."


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
    """Read files — supports txt, md, pdf, csv, json, yaml."""
    if not path:
        return "No file path provided."
    try:
        import os
        if not os.path.exists(path):
            return f"File not found: {path}"

        ext = os.path.splitext(path)[1].lower()

        if ext in [".txt", ".md", ".json", ".yaml", ".yml", ".py", ".js", ".html"]:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read(10000)
            return f"File: {path}\n\n{content}"

        elif ext == ".csv":
            import csv
            rows = []
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i > 50:
                        rows.append("... (truncated at 50 rows)")
                        break
                    rows.append(", ".join(row))
            return f"CSV file: {path}\n\n" + "\n".join(rows)

        elif ext == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(path)
                text = ""
                for page in reader.pages[:10]:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
                return f"PDF file: {path}\n\n{text[:8000]}"
            except ImportError:
                return "PDF reading requires pypdf. Install with: pip install pypdf"

        else:
            return f"Unsupported file type: {ext}. Supported: txt, md, json, yaml, py, js, html, csv, pdf"

    except Exception as e:
        logger.error(f"File reader error: {e}")
        return f"File read failed: {e}"


async def _tool_code_runner(code: str) -> str:
    """
    Run Python code in a sandboxed subprocess.
    10 second timeout. Captures stdout and stderr.
    """
    if not code:
        return "No code provided."

    blocked = ["os.system", "subprocess.call", "subprocess.run",
               "eval(", "exec(", "__import__", "open('/etc", "open('/proc"]
    for b in blocked:
        if b in code:
            return f"Blocked operation detected: '{b}'. For security, this operation is not allowed."

    import asyncio
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                     delete=False, dir="/tmp") as f:
        f.write(code)
        tmp_path = f.name

    try:
        proc = await asyncio.create_subprocess_exec(
            "python3", tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "PYTHONPATH": ""},
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=10.0
            )
        except asyncio.TimeoutError:
            proc.kill()
            return "Code execution timed out (10 second limit)."

        output = ""
        if stdout:
            output += f"Output:\n{stdout.decode()}"
        if stderr:
            output += f"\nErrors:\n{stderr.decode()}"
        if not output:
            output = "Code ran successfully with no output."

        return output

    except Exception as e:
        logger.error(f"Code runner error: {e}")
        return f"Code execution failed: {e}"
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


async def _tool_rag(query: str, sources: list[str]) -> str:
    """
    Retrieval-Augmented Generation: find relevant passages from documents.
    Reads each source file, splits into chunks, scores by keyword overlap,
    and returns the top passages. No vector DB required.
    """
    if not query:
        return "No query provided for RAG."
    if not sources:
        return "No document sources provided for RAG."

    import os

    all_chunks: list[tuple[float, str, str]] = []
    query_words = set(query.lower().split())

    for src in sources:
        try:
            if os.path.exists(src):
                ext = os.path.splitext(src)[1].lower()
                if ext == ".pdf":
                    try:
                        import pypdf
                        reader = pypdf.PdfReader(src)
                        text = "\n".join(p.extract_text() or "" for p in reader.pages[:20])
                    except ImportError:
                        text = f"[PDF reading requires pypdf: {src}]"
                elif ext in (".txt", ".md", ".rst"):
                    with open(src, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                elif ext == ".csv":
                    with open(src, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read(20000)
                else:
                    with open(src, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read(20000)
            else:
                async with httpx.AsyncClient(timeout=15.0) as h:
                    r = await h.get(src)
                    text = r.text[:20000]

            words = text.split()
            chunk_size = 300
            for i in range(0, len(words), chunk_size // 2):
                chunk_words = words[i: i + chunk_size]
                chunk = " ".join(chunk_words)
                chunk_words_lower = set(w.lower() for w in chunk_words)
                score = len(query_words & chunk_words_lower) / max(len(query_words), 1)
                all_chunks.append((score, src, chunk))

        except Exception as e:
            logger.warning(f"RAG: could not read source {src}: {e}")

    if not all_chunks:
        return f"RAG: could not read any of the provided sources: {sources}"

    top = sorted(all_chunks, key=lambda x: x[0], reverse=True)[:5]
    parts = []
    for score, src, chunk in top:
        parts.append(f"[Source: {src} | relevance: {score:.2f}]\n{chunk}")

    return "\n\n---\n\n".join(parts)


async def _tool_slack(channel: str, message: str) -> str:
    """Post a message to Slack via incoming webhook (SLACK_WEBHOOK_URL env var)."""
    import os
    if not channel or not message:
        return "Slack tool requires both 'channel' and 'message'."

    webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")
    if not webhook_url:
        return "Slack tool: SLACK_WEBHOOK_URL environment variable is not set."

    payload = {
        "channel": channel if channel.startswith("#") else f"#{channel}",
        "text": message,
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as h:
            resp = await h.post(webhook_url, json=payload)
            if resp.status_code == 200:
                return f"Message posted to Slack channel {payload['channel']}."
            return f"Slack returned HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        logger.error(f"Slack tool error: {e}")
        return f"Slack tool failed: {e}"


async def _tool_email(to: str, subject: str, body: str) -> str:
    """Send an email via SMTP (SMTP_HOST / SMTP_USER / SMTP_PASS env vars)."""
    import os, smtplib
    from email.mime.text import MIMEText

    if not to or not subject or not body:
        return "Email tool requires 'to', 'subject', and 'body'."

    smtp_host = os.getenv("SMTP_HOST", "")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_pass = os.getenv("SMTP_PASS", "")

    if not smtp_host:
        return "Email tool: SMTP_HOST environment variable is not set."

    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"]    = smtp_user
        msg["To"]      = to

        def _send_sync():
            with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
                server.starttls()
                if smtp_user and smtp_pass:
                    server.login(smtp_user, smtp_pass)
                server.send_message(msg)

        import asyncio
        await asyncio.to_thread(_send_sync)

        return f"Email sent to {to} with subject '{subject}'."
    except Exception as e:
        logger.error(f"Email tool error: {e}")
        return f"Email tool failed: {e}"


async def _tool_google_docs(doc_id: str, action: str, content: str = "") -> str:
    """
    Read from or append to a Google Doc.
    Requires GOOGLE_SERVICE_ACCOUNT_JSON env var (path to service account key file).
    """
    import os
    if not doc_id:
        return "Google Docs tool requires 'doc_id'."
    if action not in ("read", "append"):
        return "Google Docs tool: action must be 'read' or 'append'."

    sa_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    if not sa_path:
        return "Google Docs tool: GOOGLE_SERVICE_ACCOUNT_JSON environment variable is not set."

    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        creds = service_account.Credentials.from_service_account_file(
            sa_path,
            scopes=["https://www.googleapis.com/auth/documents"],
        )
        service = build("docs", "v1", credentials=creds)

        if action == "read":
            doc = service.documents().get(documentId=doc_id).execute()
            text_parts = []
            for elem in doc.get("body", {}).get("content", []):
                for para in elem.get("paragraph", {}).get("elements", []):
                    text_parts.append(para.get("textRun", {}).get("content", ""))
            return "".join(text_parts)[:8000]

        elif action == "append":
            if not content:
                return "Google Docs append requires 'content'."
            requests = [{"insertText": {"location": {"index": 1}, "text": content + "\n"}}]
            service.documents().batchUpdate(
                documentId=doc_id, body={"requests": requests}
            ).execute()
            return f"Appended content to Google Doc {doc_id}."

    except ImportError:
        return "Google Docs tool requires google-api-python-client: pip install google-api-python-client google-auth"
    except Exception as e:
        logger.error(f"Google Docs tool error: {e}")
        return f"Google Docs tool failed: {e}"


async def _tool_notion(page_id: str, action: str, content: str = "") -> str:
    """
    Read from or append to a Notion page.
    Requires NOTION_API_KEY env var.
    """
    import os
    if not page_id:
        return "Notion tool requires 'page_id'."
    if action not in ("read", "append"):
        return "Notion tool: action must be 'read' or 'append'."

    api_key = os.getenv("NOTION_API_KEY", "")
    if not api_key:
        return "Notion tool: NOTION_API_KEY environment variable is not set."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as h:
            if action == "read":
                resp = await h.get(
                    f"https://api.notion.com/v1/blocks/{page_id}/children",
                    headers=headers,
                )
                resp.raise_for_status()
                blocks = resp.json().get("results", [])
                text_parts = []
                for block in blocks:
                    btype = block.get("type", "")
                    rich_text = block.get(btype, {}).get("rich_text", [])
                    for rt in rich_text:
                        text_parts.append(rt.get("plain_text", ""))
                return "\n".join(text_parts)[:8000] or "(page is empty)"

            elif action == "append":
                if not content:
                    return "Notion append requires 'content'."
                body = {
                    "children": [
                        {
                            "object": "block",
                            "type": "paragraph",
                            "paragraph": {
                                "rich_text": [{"type": "text", "text": {"content": content}}]
                            },
                        }
                    ]
                }
                resp = await h.patch(
                    f"https://api.notion.com/v1/blocks/{page_id}/children",
                    headers=headers,
                    json=body,
                )
                resp.raise_for_status()
                return f"Appended content to Notion page {page_id}."

    except Exception as e:
        logger.error(f"Notion tool error: {e}")
        return f"Notion tool failed: {e}"


# ── All tool schemas ───────────────────────────────────────────────────────────────────────────
# Defined at module level so it's not rebuilt on every agent invocation

ALL_TOOL_SCHEMAS = {
    "web_search": {
        "name": "web_search",
        "description": "Search the web for current information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"],
        },
    },
    "code_runner": {
        "name": "code_runner",
        "description": "Execute Python code in a sandbox and return the output",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to run"}
            },
            "required": ["code"],
        },
    },
    "file_reader": {
        "name": "file_reader",
        "description": "Read a file from disk (txt, md, json, yaml, csv, pdf)",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file"}
            },
            "required": ["path"],
        },
    },
    "rag": {
        "name": "rag",
        "description": "Retrieve relevant passages from documents or URLs",
        "parameters": {
            "type": "object",
            "properties": {
                "query":   {"type": "string", "description": "Search query"},
                "sources": {"type": "array", "items": {"type": "string"},
                            "description": "File paths or URLs to search"},
            },
            "required": ["query", "sources"],
        },
    },
    "slack": {
        "name": "slack",
        "description": "Post a message to a Slack channel",
        "parameters": {
            "type": "object",
            "properties": {
                "channel": {"type": "string", "description": "Slack channel name (with or without #)"},
                "message": {"type": "string", "description": "Message text to send"},
            },
            "required": ["channel", "message"],
        },
    },
    "email": {
        "name": "email",
        "description": "Send an email via SMTP",
        "parameters": {
            "type": "object",
            "properties": {
                "to":      {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject line"},
                "body":    {"type": "string", "description": "Email body text"},
            },
            "required": ["to", "subject", "body"],
        },
    },
    "google_docs": {
        "name": "google_docs",
        "description": "Read from or append to a Google Doc",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_id":  {"type": "string", "description": "Google Doc ID from the URL"},
                "action":  {"type": "string", "enum": ["read", "append"],
                            "description": "read or append"},
                "content": {"type": "string", "description": "Content to append (required for append)"},
            },
            "required": ["doc_id", "action"],
        },
    },
    "notion": {
        "name": "notion",
        "description": "Read from or append to a Notion page",
        "parameters": {
            "type": "object",
            "properties": {
                "page_id": {"type": "string", "description": "Notion page ID"},
                "action":  {"type": "string", "enum": ["read", "append"],
                            "description": "read or append"},
                "content": {"type": "string", "description": "Content to append (required for append)"},
            },
            "required": ["page_id", "action"],
        },
    },
}


# ── Agent loop ────────────────────────────────────────────────────────────────────────────────

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

    # ── Pick model ────────────────────────────────────────────────────────────────────────────
    if model_preference == "auto":
        task_type, config_name, complexity = route(user_message, is_agent=True)
    elif model_preference == "fast":
        config_name = "fast"
    else:
        config_name = "smart"

    logger.info(f"Agent '{agent_name}' using model config: {config_name}")

    # ── Ensure model is warm ──────────────────────────────────────────────────────────────────────────
    state = get_state(config_name)
    if state == ModelState.COLD:
        triggered = await spin_up(config_name)
        if not triggered:
            return {"error": f"Could not spin up model '{config_name}'"}
        warm = await wait_until_warm(config_name, timeout=300)
        if not warm:
            return {"error": f"Model '{config_name}' timed out during warm-up"}

    # ── Get actual model name from config ─────────────────────────────────────────────────────────────────────────────
    from scheduler import _get_model_config
    mc_spec = _get_model_config(config_name)
    actual_model = mc_spec.get("modelName", config_name) if mc_spec else config_name
    model_url = f"http://peag-{config_name}.ai.svc.cluster.local:{OLLAMA_PORT}"

    # ── Build tool definitions ─────────────────────────────────────────────────────────────────────────────────
    # Check tool_registry for per-tool model overrides
    try:
        from tool_registry import get_model_for_tool
    except ImportError:
        get_model_for_tool = lambda t: config_name  # noqa

    # Models that don't support tool calling
    NON_TOOL_MODELS = {"tinyllama", "phi3", "phi3:mini", "phi3:medium"}
    supports_tools = (actual_model not in NON_TOOL_MODELS)

    tool_definitions = []
    for tool_name in tools:
        if tool_name in ALL_TOOL_SCHEMAS:
            tool_definitions.append({
                "type": "function",
                "function": ALL_TOOL_SCHEMAS[tool_name],
            })
        else:
            logger.warning(f"Agent '{agent_name}' lists unknown tool '{tool_name}' — skipping")

    # ── Build messages ────────────────────────────────────────────────────────────────────────────────────
    messages = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        messages.extend(conversation_history[-10:])
    messages.append({"role": "user", "content": user_message})

    # ── Agent loop (max 5 iterations to prevent infinite loops) ───────────────────────────────────────────────────────────────────────────────────
    max_iterations = 5
    iteration = 0

    async with httpx.AsyncClient(timeout=120.0) as http:
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Agent loop iteration {iteration}")
            update_last_request(config_name)

            payload = {
                "model": actual_model,
                "messages": messages,
                "stream": False,
            }
            if supports_tools and tool_definitions:
                payload["tools"] = tool_definitions

            resp = await http.post(
                f"{model_url}/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            message = choice["message"]

            # ── Tool call? Execute and loop back ────────────────────────────────────────────────────────────────────────────────
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

            # ── Final response ─────────────────────────────────────────────────────────────────────────────────────────────
            final_content = message.get("content", "")
            logger.info(f"Agent '{agent_name}' completed in {iteration} iteration(s)")

            return {
                "agent": agent_display_name,
                "model_used": config_name,
                "actual_model": actual_model,
                "iterations": iteration,
                "tools_available": tools,
                "response": final_content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    return {"error": "Agent loop exceeded maximum iterations"}