import logging

logger = logging.getLogger(__name__)

# ── Task classification rules ─────────────────────────────────────────────────
# Keywords are matched against the prompt (case-insensitive).
# First match wins. Default fallback at the bottom.

RULES = [
    {
        "task": "code",
        "model": "llama3",
        "keywords": [
            "code", "function", "script", "debug", "error", "bug",
            "program", "class", "algorithm", "implement", "refactor",
            "python", "javascript", "sql", "bash", "library"
        ]
    },
    {
        "task": "creative",
        "model": "llama3",
        "keywords": [
            "write", "story", "poem", "creative", "essay", "blog",
            "draft", "compose", "narrative", "fiction", "describe"
        ]
    },
    {
        "task": "summary",
        "model": "phi3",
        "keywords": [
            "summarize", "summary", "tldr", "shorten", "condense",
            "brief", "overview", "recap", "outline", "key points"
        ]
    },
    {
        "task": "question",
        "model": "phi3",
        "keywords": [
            "what is", "what are", "who is", "where is", "when did",
            "how does", "why is", "explain", "define", "tell me"
        ]
    },
]

DEFAULT_MODEL = "phi3"


def classify(prompt: str) -> tuple[str, str]:
    """
    Classify a prompt and return (task_type, model_name).
    Checks keywords in order — first match wins.
    Falls back to DEFAULT_MODEL if nothing matches.
    """
    prompt_lower = prompt.lower()

    for rule in RULES:
        for keyword in rule["keywords"]:
            if keyword in prompt_lower:
                logger.info(
                    f"Classified as '{rule['task']}' "
                    f"(matched: '{keyword}') → model: {rule['model']}"
                )
                return rule["task"], rule["model"]

    logger.info(f"No match found → default model: {DEFAULT_MODEL}")
    return "general", DEFAULT_MODEL


def extract_prompt(messages: list[dict]) -> str:
    """Extract the last user message from an OpenAI-format messages array."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                # Handle multimodal content
                return " ".join(
                    part.get("text", "")
                    for part in content
                    if part.get("type") == "text"
                )
            return content
    return ""
