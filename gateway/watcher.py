import asyncio
import logging
import time
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from state import ModelState, get_state, get_last_request
from scheduler import scale_to_zero

logger = logging.getLogger(__name__)

NAMESPACE = "ai"
CHECK_INTERVAL_SECONDS = 30  # how often the watcher runs


def _load_kube_config():
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()


def _get_all_model_configs() -> list[dict]:
    """Fetch all ModelConfig CRDs from the cluster."""
    try:
        _load_kube_config()
        custom = client.CustomObjectsApi()
        result = custom.list_namespaced_custom_object(
            group="peag.io",
            version="v1alpha1",
            namespace=NAMESPACE,
            plural="modelconfigs",
        )
        return result.get("items", [])
    except ApiException as e:
        logger.error(f"Failed to list ModelConfigs: {e}")
        return []


def _is_deployment_running(model_name: str) -> bool:
    """Check if a model deployment actually exists and has ready replicas."""
    try:
        _load_kube_config()
        apps_v1 = client.AppsV1Api()
        dep = apps_v1.read_namespaced_deployment(
            name=f"peag-{model_name}", namespace=NAMESPACE
        )
        return (dep.status.ready_replicas or 0) >= 1
    except ApiException as e:
        if e.status == 404:
            return False
        logger.error(f"Error checking deployment for {model_name}: {e}")
        return False


async def _check_model(model_name: str, idle_ttl: int) -> None:
    """Check a single model and scale it down if idle past TTL."""
    state = get_state(model_name)

    # Only check models that are warm — cold/warming models are not running
    if state != ModelState.WARM:
        # But if state says warm and no deployment exists, fix the state
        if state == ModelState.WARM and not _is_deployment_running(model_name):
            logger.warning(f"{model_name} state is warm but no deployment found — correcting to cold")
            from state import set_state
            set_state(model_name, ModelState.COLD)
        return

    last_request = get_last_request(model_name)

    if last_request is None:
        # Warm but never received a request — scale down immediately
        logger.info(f"{model_name} is warm but has no request history — scaling to zero")
        await scale_to_zero(model_name)
        return

    idle_seconds = time.time() - last_request

    if idle_seconds >= idle_ttl:
        logger.info(
            f"{model_name} has been idle for {idle_seconds:.0f}s "
            f"(TTL: {idle_ttl}s) — scaling to zero"
        )
        await scale_to_zero(model_name)
    else:
        remaining = idle_ttl - idle_seconds
        logger.debug(f"{model_name} is active — {remaining:.0f}s until idle TTL")


async def run_watcher() -> None:
    """Main watcher loop — runs forever, checking all models every 30 seconds."""
    logger.info("PEAG idle watcher started")

    while True:
        try:
            configs = _get_all_model_configs()

            if not configs:
                logger.debug("No ModelConfigs found — nothing to watch")
            else:
                for mc in configs:
                    model_name = mc["metadata"]["name"]
                    idle_ttl = mc.get("spec", {}).get("idleTTLSeconds", 300)
                    await _check_model(model_name, idle_ttl)

        except Exception as e:
            # Never let the watcher crash — log and keep going
            logger.error(f"Watcher loop error: {e}")

        await asyncio.sleep(CHECK_INTERVAL_SECONDS)
