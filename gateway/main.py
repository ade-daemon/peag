import asyncio
import logging
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from state import (
    ModelState, get_state, set_state,
    update_last_request, increment_pending, decrement_pending
)
from scheduler import spin_up, wait_until_warm, _get_model_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PEAG Gateway",
    description="Dynamic GPU/CPU orchestrator for local LLMs",
    version="0.1.0",
)

# Where the model pods are reachable inside the cluster
# peag-{model_name}.ai.svc.cluster.local is the DNS pattern we use
OLLAMA_PORT = 11434


def model_url(model_name: str) -> str:
    return f"http://peag-{model_name}.ai.svc.cluster.local:{OLLAMA_PORT}"


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "peag-gateway"}


# ── Model state endpoint — used by dashboards and the idle watcher ────────────
@app.get("/models/{model_name}/state")
async def model_state(model_name: str):
    state = get_state(model_name)
    return {"model": model_name, "state": state.value}


# ── Main inference endpoint — OpenAI-compatible ───────────────────────────────
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model_name = body.get("model")

    if not model_name:
        raise HTTPException(status_code=400, detail="model field is required")

    logger.info(f"Request for model: {model_name}")

    state = get_state(model_name)
    increment_pending(model_name)
    update_last_request(model_name)

    try:
        # ── WARM: proxy straight through ─────────────────────────────────────
        if state == ModelState.WARM:
            return await _proxy_request(model_name, body, request)

        # ── WARMING: another request already triggered spin-up, just wait ────
        if state == ModelState.WARMING:
            logger.info(f"{model_name} is warming — waiting for readiness")
            warm = await wait_until_warm(model_name)
            if not warm:
                raise HTTPException(
                    status_code=503,
                    detail=f"Model {model_name} did not become ready in time. Try again shortly.",
                )
            return await _proxy_request(model_name, body, request)

        # ── COLD: spin it up ─────────────────────────────────────────────────
        if state == ModelState.COLD:
            logger.info(f"{model_name} is cold — triggering spin-up")
            triggered = await spin_up(model_name)
            if not triggered:
                raise HTTPException(
                    status_code=404,
                    detail=f"No ModelConfig found for model: {model_name}",
                )

            # Default cold-start behaviour: queue and hold
            # TODO: read coldStartBehaviour from ModelConfig CRD and branch:
            #   "queue" → wait here (current behaviour)
            #   "sse"   → stream warming status
            #   "503"   → return immediately with Retry-After
            warm = await wait_until_warm(model_name, timeout=300)
            if not warm:
                raise HTTPException(
                    status_code=503,
                    detail=f"Model {model_name} timed out during cold start.",
                )
            return await _proxy_request(model_name, body, request)

    finally:
        decrement_pending(model_name)


# ── /api/tags — return all available models across all ModelConfigs ────────────
@app.get("/api/tags")
async def api_tags():
    """
    Open WebUI calls this to list available models.
    We return all models defined in ModelConfigs so the UI shows them.
    """
    from scheduler import _get_model_config
    from kubernetes import client, config as kube_config

    try:
        kube_config.load_incluster_config()
    except Exception:
        kube_config.load_kube_config()

    custom = client.CustomObjectsApi()
    configs = custom.list_namespaced_custom_object(
        group="peag.io",
        version="v1alpha1",
        namespace="ai",
        plural="modelconfigs",
    )

    models = [
        {
            "name": "peag-auto:latest",
            "model": "peag-auto:latest",
            "modified_at": "2026-01-01T00:00:00Z",
            "size": 0,
            "digest": "",
            "details": {
                "family": "peag",
                "parameter_size": "auto",
                "quantization_level": "auto",
            }
        }
    ]

    for mc in configs.get("items", []):
        spec = mc.get("spec", {})
        name = mc["metadata"]["name"]
        model_id = spec.get("modelName", name)
        models.append({
            "name": f"{model_id}:latest",
            "model": f"{model_id}:latest",
            "modified_at": "2026-01-01T00:00:00Z",
            "size": 0,
            "digest": "",
            "details": {
                "family": "llama",
                "parameter_size": "unknown",
                "quantization_level": "unknown",
            }
        })

    return {"models": models}


@app.get("/api/version")
async def api_version():
    """Open WebUI calls this to check Ollama version."""
    return {"version": "0.1.0"}


# ── Ollama native API passthrough (for Open WebUI compatibility) ───────────────
@app.api_route("/api/{path:path}", methods=["GET", "POST", "DELETE"])
async def ollama_passthrough(path: str, request: Request):
    """
    Open WebUI calls Ollama's native /api/* endpoints.
    This passthrough lets Open WebUI work through the PEAG gateway
    without any changes to Open WebUI's config.
    """
    body = await request.body()

    # Try to extract model name from body
    model_name = None
    try:
        import json
        parsed = json.loads(body)
        model_name = parsed.get("model")
    except Exception:
        pass

    if model_name:
        # Strip :latest suffix
        model_name = model_name.replace(":latest", "")
        
        # peag-auto — route through task intelligence
        if model_name == "peag-auto":
            body_json = json.loads(body)
            messages = body_json.get("messages", [])
            from task_intelligence import route, extract_prompt
            prompt = extract_prompt(messages)
            task_type, config_name, complexity = route(prompt, is_agent=False)
            mc_spec = _get_model_config(config_name)
            actual_model = mc_spec.get("modelName", config_name) if mc_spec else config_name
            body_json["model"] = actual_model
            logger.info(f"peag-auto: task={task_type} complexity={complexity} → {config_name} ({actual_model})")
            state = get_state(config_name)
            if state == ModelState.COLD:
                await spin_up(config_name)
                await wait_until_warm(config_name)
            update_last_request(config_name)
            body = json.dumps(body_json).encode()
            target_url = f"{model_url(config_name)}/api/{path}"
            http = httpx.AsyncClient(timeout=120.0)
            async def stream_auto():
                try:
                    async with http.stream(method=request.method, url=target_url, content=body, headers={"Content-Type": "application/json"}) as resp:
                        async for chunk in resp.aiter_bytes():
                            yield chunk
                finally:
                    await http.aclose()
            return StreamingResponse(stream_auto(), media_type="application/x-ndjson")
        
        # Find the ModelConfig that owns this model
        from scheduler import find_config_by_model_name
        config_name = find_config_by_model_name(model_name)
        if config_name:
            state = get_state(config_name)
            if state == ModelState.COLD:
                await spin_up(config_name)
                await wait_until_warm(config_name)
            update_last_request(config_name)
            target_url = f"{model_url(config_name)}/api/{path}"
        else:
            target_url = f"http://ollama.ai.svc.cluster.local:{OLLAMA_PORT}/api/{path}"
    else:
        # No model specified — route to the static Ollama deployment
        target_url = f"http://ollama.ai.svc.cluster.local:{OLLAMA_PORT}/api/{path}"

    http = httpx.AsyncClient(timeout=120.0)
    
    async def stream_ollama():
        try:
            async with http.stream(
                method=request.method,
                url=target_url,
                content=body,
                headers={"Content-Type": "application/json"},
            ) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk
        finally:
            await http.aclose()

    return StreamingResponse(
        stream_ollama(),
        media_type="application/x-ndjson",
    )


# ── Internal proxy helper ─────────────────────────────────────────────────────
async def _proxy_request(model_name: str, body: dict, request: Request):
    """Forward the request to the model's pod and stream the response back."""
    target = f"{model_url(model_name)}/v1/chat/completions"
    streaming = body.get("stream", False)

    async with httpx.AsyncClient(timeout=120.0) as http:
        if streaming:
            async def stream_response():
                async with http.stream("POST", target, json=body) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
            )
        else:
            resp = await http.post(target, json=body)
            return JSONResponse(content=resp.json(), status_code=resp.status_code)


# ── Start idle watcher and agent scheduler as background tasks on startup ───
@app.on_event("startup")
async def start_watcher():
    from watcher import run_watcher
    from agent_scheduler import run_agent_scheduler
    asyncio.create_task(run_watcher())
    asyncio.create_task(run_agent_scheduler())
    logger.info("Idle watcher and agent scheduler started")


# ── PEAG custom Prometheus metrics ───────────────────────────────────────────
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
import time as time_module

# Track every request by model and whether it was a cold or warm hit
REQUEST_COUNTER = Counter(
    "peag_requests_total",
    "Total requests handled by PEAG",
    ["model", "start_type"]  # start_type: cold | warm | warming
)

# Track cold-start latency — the key metric for the demo
COLD_START_HISTOGRAM = Histogram(
    "peag_cold_start_seconds",
    "Time taken to warm a model from cold",
    ["model"],
    buckets=[5, 10, 20, 30, 45, 60, 90, 120]
)

# Track how many models are currently warm
WARM_MODELS_GAUGE = Gauge(
    "peag_warm_models_total",
    "Number of models currently warm"
)

# Expose /metrics endpoint for Prometheus to scrape
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# ── Auto-routing: classify prompt and select model automatically ──────────────
@app.post("/v1/chat/completions/auto")
async def auto_chat_completions(request: Request):
    """
    OpenAI-compatible endpoint that automatically selects the right model
    based on the prompt content. Clients don't need to specify a model.
    """
    from task_intelligence import route, extract_prompt

    body = await request.json()
    messages = body.get("messages", [])
    prompt = extract_prompt(messages)

    if not prompt:
        raise HTTPException(status_code=400, detail="No user message found")

    task_type, model_name, complexity = route(prompt, is_agent=False)
    logger.info(f"Task intelligence: type={task_type} complexity={complexity} model={model_name}")

    # Resolve actual model identifier from ModelConfig
    mc_spec = _get_model_config(model_name)
    actual_model = mc_spec.get("modelName", model_name) if mc_spec else model_name
    body["model"] = actual_model
    logger.info(f"Resolved model: {model_name} → {actual_model}")
    logger.info(f"Auto-routing: task={task_type} → model={actual_model}")

    # Add classification metadata to the request for metrics
    REQUEST_COUNTER.labels(model=actual_model, start_type="auto").inc()
    update_last_request(model_name)
    increment_pending(model_name)

    try:
        state = get_state(model_name)

        if state == ModelState.COLD:
            await spin_up(model_name)
            warm = await wait_until_warm(model_name, timeout=300)
            if not warm:
                raise HTTPException(
                    status_code=503,
                    detail=f"Model {model_name} timed out during cold start"
                )
        elif state == ModelState.WARMING:
            warm = await wait_until_warm(model_name)
            if not warm:
                raise HTTPException(status_code=503, detail="Model not ready")

        return await _proxy_request(model_name, body, request)

    finally:
        decrement_pending(model_name)




@app.get("/")
async def root():
    return {
        "service": "PEAG Gateway",
        "version": "0.1.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "models": "/api/tags",
            "chat": "/v1/chat/completions",
            "auto_chat": "/v1/chat/completions/auto",
            "docs": "/docs"
        }
    }


# ── Agent endpoints ───────────────────────────────────────────────────────────

@app.get("/v1/agents")
async def list_agents():
    """List all configured agents."""
    from agent_executor import _get_all_agent_configs
    configs = _get_all_agent_configs()
    agents = []
    for ac in configs:
        spec = ac.get("spec", {})
        agents.append({
            "id": ac["metadata"]["name"],
            "name": spec.get("name", ac["metadata"]["name"]),
            "model": spec.get("model", "auto"),
            "tools": spec.get("tools", []),
            "schedule": spec.get("schedule"),
        })
    return {"agents": agents}


@app.post("/v1/agents/{agent_name}/run")
async def run_agent_endpoint(agent_name: str, request: Request):
    """Run an agent with a user message."""
    from agent_executor import run_agent
    body = await request.json()
    user_message = body.get("message", "")
    history = body.get("history", [])

    if not user_message:
        raise HTTPException(status_code=400, detail="message field is required")

    logger.info(f"Running agent: {agent_name}")
    result = await run_agent(agent_name, user_message, history)
    return result


@app.get("/v1/agents/{agent_name}")
async def get_agent(agent_name: str):
    """Get details of a specific agent."""
    from agent_executor import _get_agent_config
    spec = _get_agent_config(agent_name)
    if not spec:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    return {"agent": agent_name, "spec": spec}
