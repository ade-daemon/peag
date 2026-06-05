import redis
import logging
from enum import Enum

logger = logging.getLogger(__name__)

# ── Model lifecycle states ────────────────────────────────────────────────────
class ModelState(str, Enum):
    COLD    = "cold"      # not running, needs to be spun up
    WARMING = "warming"   # spin-up triggered, not ready yet
    WARM    = "warm"      # running and ready to serve requests

# ── Redis connection ──────────────────────────────────────────────────────────
# redis.ai.svc.cluster.local is the DNS name of our Redis service inside the cluster
redis_client = redis.Redis(
    host="redis.ai.svc.cluster.local",
    port=6379,
    decode_responses=True,  # return strings not bytes
)

# How long a model stays "warm" in Redis before we consider it cold again.
# The idle watcher handles actual pod scale-down — this TTL just prevents
# Redis from holding stale warm states if the watcher misses something.
WARM_TTL_SECONDS = 700  # 11 minutes — slightly longer than the longest model idleTTL


def get_state(model_name: str) -> ModelState:
    """Return the current state of a model. Defaults to COLD if not in Redis."""
    try:
        value = redis_client.get(f"model:state:{model_name}")
        if value is None:
            return ModelState.COLD
        return ModelState(value)
    except Exception as e:
        logger.error(f"Redis get_state error for {model_name}: {e}")
        return ModelState.COLD  # safe default — treat unknown as cold


def set_state(model_name: str, state: ModelState) -> None:
    """Write a model's state to Redis."""
    try:
        key = f"model:state:{model_name}"
        if state == ModelState.WARM:
            # Warm states expire automatically — safety net against stale data
            redis_client.setex(key, WARM_TTL_SECONDS, state.value)
        else:
            # Cold and warming states don't expire — they're updated explicitly
            redis_client.set(key, state.value)
        logger.info(f"Model {model_name} state → {state.value}")
    except Exception as e:
        logger.error(f"Redis set_state error for {model_name}: {e}")


def update_last_request(model_name: str) -> None:
    """Record the time of the last request — used by the idle watcher."""
    try:
        import time
        redis_client.set(f"model:last_request:{model_name}", str(time.time()))
    except Exception as e:
        logger.error(f"Redis update_last_request error for {model_name}: {e}")


def get_last_request(model_name: str) -> float | None:
    """Return the timestamp of the last request, or None if never requested."""
    try:
        value = redis_client.get(f"model:last_request:{model_name}")
        return float(value) if value else None
    except Exception as e:
        logger.error(f"Redis get_last_request error for {model_name}: {e}")
        return None


def increment_pending(model_name: str) -> int:
    """Increment the pending request counter. Used by KEDA for scaling decisions."""
    try:
        count = redis_client.incr(f"model:pending:{model_name}")
        redis_client.expire(f"model:pending:{model_name}", 60)  # auto-clear after 60s
        return count
    except Exception as e:
        logger.error(f"Redis increment_pending error for {model_name}: {e}")
        return 0


def decrement_pending(model_name: str) -> None:
    """Decrement the pending request counter once a request is handled."""
    try:
        redis_client.decr(f"model:pending:{model_name}")
    except Exception as e:
        logger.error(f"Redis decrement_pending error for {model_name}: {e}")


def get_pending(model_name: str) -> int:
    """Return the current number of pending requests for a model."""
    try:
        value = redis_client.get(f"model:pending:{model_name}")
        return int(value) if value else 0
    except Exception as e:
        logger.error(f"Redis get_pending error for {model_name}: {e}")
        return 0
