# PEAG — Private Enterprise AI Gateway

> Self-hosted AI infrastructure that treats local LLMs like serverless functions.

PEAG is a Kubernetes-native orchestrator for private AI. No model runs unless someone needs it. The right model spins up on the right hardware when a request arrives, and scales back to absolute zero when idle. Your data never leaves your building.

---

## The problem

Companies running internal AI tools pay for GPU and CPU compute around the clock — even at 2am when nobody is using it. Most enterprise AI deployments run at under 40% utilisation. The rest is waste.

PEAG fixes that.

---

## How it works

```
Request arrives
      ↓
Task intelligence reads the prompt
Scores complexity · classifies task type
      ↓
Routes to the right model automatically
  quick question → fast model (CPU)
  complex code   → smart model (GPU)
      ↓
Model spins up if cold (loads from local disk)
      ↓
Response delivered
      ↓
Model scales to zero after idle TTL
```

No idle compute. No wasted GPU hours. No paying for a model that's just sitting there.

---

## Install

```bash
curl -fsSL https://peag.io/install | bash
```

The installer will:
- Install k3s (lightweight Kubernetes)
- Ask which AI models you want (19 options)
- Pull selected models
- Start Open WebUI (chat interface)
- Start PEAG Agent Studio
- Give you a URL to share with your team

### Requirements

- Ubuntu 20.04+ or Debian 11+
- 8GB+ RAM (16GB recommended)
- 20GB+ free disk space
- GPU optional (NVIDIA recommended for large models)

---

## Models

Choose at install time. Pull more anytime with `peag pull <model>`.

| Model | Size | Hardware | Best for |
|-------|------|----------|----------|
| TinyLlama 1B | 637MB | CPU | Basic Q&A, very fast |
| Phi-3 Mini | 2.2GB | CPU | Smart, efficient — recommended |
| Gemma2 2B | 1.6GB | CPU | Google's compact model |
| Llama 3.2 8B | 4.7GB | GPU | High quality — recommended |
| Mistral 7B | 4.1GB | GPU | Excellent reasoning and code |
| DeepSeek-Coder 6.7B | 3.8GB | GPU | Best open source code model |
| DeepSeek-R1 7B | 4.7GB | GPU | Strong reasoning |
| Llama 3.1 70B | 40GB | GPU | Near GPT-4 quality |
| + 11 more | — | — | See installer for full list |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Clients                          │
│         Open WebUI · Claude Code · Cursor           │
│              Any OpenAI-compatible tool             │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              PEAG Gateway (FastAPI)                 │
│         OpenAI-compatible API · /v1/...             │
│                                                     │
│  Task Intelligence    │    Agent Runtime            │
│  ─────────────────    │    ─────────────            │
│  Complexity scorer    │    AgentConfig CRDs         │
│  Task classifier      │    Tool executor            │
│  Model router         │    Cron scheduler           │
└──────┬───────────────────────────┬──────────────────┘
       │                           │
┌──────▼──────┐           ┌────────▼───────┐
│    Redis    │           │   Kubernetes   │
│ State cache │           │  Dynamic pods  │
│  tool map   │           │  Scale to zero │
└─────────────┘           └────────────────┘
```

### Components

**PEAG Gateway** — FastAPI application that intercepts every request. Exposes an OpenAI-compatible API so any existing tool works without changes.

**Task Intelligence** — reads each prompt, scores complexity (1-10), classifies the task type (retrieval, code, reasoning, creative, summary), and routes to the appropriate model automatically.

**ModelConfig CRD** — custom Kubernetes resource that defines each model: backend (Ollama/vLLM), hardware class (GPU/CPU), idle TTL, cold-start behaviour, and resource limits.

**AgentConfig CRD** — custom Kubernetes resource that defines agents: system prompt, tools, schedule, and output channel.

**Idle Watcher** — background loop that checks every model against its idle TTL and scales pods and services to zero when the threshold is exceeded.

**Agent Scheduler** — background loop that fires scheduled agents based on cron expressions and delivers output to Slack, email, or webhook.

**KEDA ScaledObjects** — reactive scaling based on request queue depth. Pods scale up under load and back to zero when idle.

**Persistent Volume** — model weights are stored on a PVC so cold starts load from local disk in seconds rather than re-downloading from the internet.

---

## Agents

PEAG includes a full agent framework. Agents can:

- Search the web (DuckDuckGo, no API key required)
- Execute Python code in a sandbox
- Read files (txt, md, csv, pdf, json)
- Send Slack messages
- Send emails
- POST to webhooks
- Run on a cron schedule automatically

### Create an agent

**Natural language (via Agent Studio):**
> "Search the web every morning and post AI news to #research on Slack"

PEAG generates the AgentConfig automatically.

**Manual (YAML):**

```yaml
apiVersion: peag.io/v1alpha1
kind: AgentConfig
metadata:
  name: researcher
  namespace: ai
spec:
  name: "Research Agent"
  model: smart
  systemPrompt: |
    You are a research assistant. Search the web for current information
    and summarise key findings with sources.
  tools:
    - web_search
    - file_reader
  memory:
    enabled: true
    maxMessages: 20
```

```bash
kubectl apply -f researcher-agent.yaml
```

### Run an agent

```bash
curl -X POST http://your-peag:8000/v1/agents/researcher/run \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the latest developments in private AI?"}'
```

---

## OpenAI-compatible tools

PEAG works with any tool built for the OpenAI API:

**Claude Code:**
```bash
export ANTHROPIC_BASE_URL=http://your-peag:8000
export ANTHROPIC_API_KEY=local
claude
```

**Cursor:**
- Settings → Models → Override OpenAI Base URL → `http://your-peag:8000`

**Continue.dev:**
```json
{
  "models": [{
    "provider": "openai",
    "model": "deepseek-coder",
    "apiBase": "http://your-peag:8000",
    "apiKey": "local"
  }]
}
```

**Python SDK:**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://your-peag:8000",
    api_key="local"
)

response = client.chat.completions.create(
    model="peag-auto",
    messages=[{"role": "user", "content": "Hello"}]
)
```

---

## PEAG CLI

```bash
peag pull mistral        # Pull a new model
peag pull codellama      # Pull code specialist
peag list                # List installed models
peag status              # Show running services
peag start               # Start PEAG
peag stop                # Stop PEAG
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/tags` | List available models |
| POST | `/v1/chat/completions` | OpenAI-compatible chat |
| POST | `/v1/chat/completions/auto` | Auto-routed chat |
| GET | `/v1/agents` | List all agents |
| POST | `/v1/agents/build` | Create agent from description |
| POST | `/v1/agents/{name}/run` | Run an agent |
| DELETE | `/v1/agents/{name}` | Delete an agent |
| GET | `/v1/tools` | List available tools |
| PUT | `/v1/tools/{name}/assign` | Assign tool to model |
| GET | `/models/{name}/state` | Get model state |

---

## Stack

- **Kubernetes** — k3s for on-premises, EKS/AKS for cloud
- **FastAPI** — gateway and agent API
- **Ollama** — model serving backend
- **Redis** — model state cache and tool assignments
- **KEDA** — queue-depth based autoscaling
- **Prometheus + Grafana** — metrics and dashboards
- **Open WebUI** — chat interface

---

## Who this is for

**Law firms** — client communications cannot leave the building.

**Healthcare** — patient data and clinical notes are subject to strict compliance rules.

**Financial services** — trading strategies and client portfolios deserve privacy.

**Government** — classified information cannot be processed on third-party servers.

**Any company** — if your team uses AI daily, PEAG costs less than per-seat subscriptions within months.

---

## License

MIT — free to use, modify, and distribute.

---

## Links

- Website: [peag.io](https://peag.io)
- GitHub: [github.com/ade-daemon/peag](https://github.com/ade-daemon/peag)
