import re
import logging

logger = logging.getLogger(__name__)

# ── Task type definitions ─────────────────────────────────────────────────────

TASK_RULES = [
    {
        "task": "code",
        "model": "smart",
        "priority": 1,
        "keywords": [
            "code", "function", "script", "debug", "error", "bug",
            "program", "class", "algorithm", "implement", "refactor",
            "python", "javascript", "typescript", "sql", "bash", "rust",
            "library", "framework", "deploy", "dockerfile", "kubernetes",
            "regex", "loop", "recursion", "async", "api endpoint"
        ]
    },
    {
        "task": "reasoning",
        "model": "smart",
        "priority": 2,
        "keywords": [
            "analyze", "analyse", "compare", "evaluate", "critique",
            "argue", "debate", "pros and cons", "trade-off", "tradeoff",
            "strategy", "plan", "design", "architect", "decision",
            "why does", "how would", "what would happen if", "implications"
        ]
    },
    {
        "task": "creative",
        "model": "smart",
        "priority": 3,
        "keywords": [
            "write a", "draft", "compose", "story", "essay", "blog",
            "article", "poem", "creative", "narrative", "persuasive",
            "proposal", "report", "letter", "email template"
        ]
    },
    {
        "task": "summary",
        "model": "fast",
        "priority": 4,
        "keywords": [
            "summarize", "summarise", "summary", "tldr", "shorten",
            "condense", "brief", "overview", "recap", "key points",
            "main points", "in short"
        ]
    },
    {
        "task": "retrieval",
        "model": "fast",
        "priority": 5,
        "keywords": [
            "what is", "what are", "who is", "where is", "when did",
            "when was", "how does", "define", "meaning of", "tell me about",
            "explain what", "what does"
        ]
    },
    {
        "task": "extraction",
        "model": "fast",
        "priority": 6,
        "keywords": [
            "extract", "find all", "list all", "pull out", "identify",
            "classify", "categorize", "tag", "label", "parse"
        ]
    },
]

DEFAULT_MODEL = "fast"


# ── Complexity scorer ─────────────────────────────────────────────────────────

def score_complexity(prompt: str) -> int:
    """
    Score prompt complexity from 1 (trivial) to 10 (very complex).
    Combines length, sentence count, and vocabulary signals.
    """
    score = 1

    words = prompt.split()
    word_count = len(words)

    # Length contribution
    if word_count > 200: score += 4
    elif word_count > 100: score += 3
    elif word_count > 50:  score += 2
    elif word_count > 20:  score += 1

    # Sentence count
    sentences = len(re.findall(r'[.!?]+', prompt))
    if sentences > 5: score += 2
    elif sentences > 2: score += 1

    # Multi-step signals
    multi_step_words = ["first", "then", "finally", "additionally", "furthermore",
                        "step", "steps", "multiple", "several", "various", "each"]
    if any(w in prompt.lower() for w in multi_step_words):
        score += 1

    # Technical vocabulary signals
    technical_words = ["implement", "architecture", "algorithm", "optimize",
                       "performance", "scalable", "distributed", "concurrent",
                       "asynchronous", "latency", "throughput", "trade-off"]
    technical_count = sum(1 for w in technical_words if w in prompt.lower())
    score += min(technical_count, 2)

    return min(score, 10)


# ── Main routing function ─────────────────────────────────────────────────────

def route(prompt: str, is_agent: bool = False) -> tuple[str, str, int]:
    """
    Route a prompt to the right model.
    Returns (task_type, model_name, complexity_score).

    Routing logic:
    - Agents start at smart, downgrade only if complexity < 3
    - Chat: classify task type + score complexity
    - High complexity always escalates to smart
    - Low complexity retrieval/summary/extraction stays fast
    """
    prompt_lower = prompt.lower()
    complexity = score_complexity(prompt)

    # Agent mode — start smart, downgrade only for trivial tasks
    if is_agent:
        if complexity < 3:
            logger.info(f"Agent route: complexity={complexity} → fast (trivial task)")
            return "retrieval", "fast", complexity
        logger.info(f"Agent route: complexity={complexity} → smart (agent default)")
        return "agent_task", "smart", complexity

    # High complexity always goes to smart regardless of task type
    if complexity >= 7:
        logger.info(f"Complexity escalation: score={complexity} → smart")
        return "complex", "smart", complexity

    # Task type matching
    matched_task = None
    matched_model = None
    best_priority = 999

    for rule in TASK_RULES:
        for keyword in rule["keywords"]:
            if keyword in prompt_lower:
                if rule["priority"] < best_priority:
                    best_priority = rule["priority"]
                    matched_task = rule["task"]
                    matched_model = rule["model"]
                break

    # Medium complexity (4-6) with smart task type → always smart
    if matched_model == "smart":
        logger.info(f"Task route: task={matched_task} complexity={complexity} → smart")
        return matched_task, "smart", complexity

    # Low complexity fast task
    if matched_task and matched_model == "fast":
        logger.info(f"Task route: task={matched_task} complexity={complexity} → fast")
        return matched_task, "fast", complexity

    # No match — use complexity to decide
    if complexity >= 4:
        logger.info(f"No match: complexity={complexity} → smart")
        return "general", "smart", complexity

    logger.info(f"No match: complexity={complexity} → fast (default)")
    return "general", DEFAULT_MODEL, complexity


def extract_prompt(messages: list[dict]) -> str:
    """Extract the last user message from OpenAI-format messages array."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                return " ".join(
                    part.get("text", "")
                    for part in content
                    if part.get("type") == "text"
                )
            return content
    return ""
