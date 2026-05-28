import logging
import asyncio
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from state import ModelState, set_state

logger = logging.getLogger(__name__)

NAMESPACE = "ai"


def _load_kube_config():
    """Load kubeconfig — tries in-cluster first, falls back to local kubeconfig."""
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()


def _get_model_config(model_name: str) -> dict | None:
    """Read the ModelConfig CRD for a given model from Kubernetes."""
    try:
        _load_kube_config()
        custom = client.CustomObjectsApi()
        obj = custom.get_namespaced_custom_object(
            group="peag.io",
            version="v1alpha1",
            namespace=NAMESPACE,
            plural="modelconfigs",
            name=model_name,
        )
        return obj.get("spec", {})
    except ApiException as e:
        if e.status == 404:
            logger.warning(f"No ModelConfig found for {model_name}")
        else:
            logger.error(f"Error reading ModelConfig for {model_name}: {e}")
        return None


def _build_deployment(model_name: str, spec: dict) -> client.V1Deployment:
    """Build a Kubernetes Deployment object for a model pod."""
    hardware = spec.get("hardware", "cpu")
    backend  = spec.get("backend", "ollama")
    resources = spec.get("resources", {})
    model_id = spec.get("modelName", model_name)

    req = resources.get("requests", {"cpu": "500m", "memory": "2Gi"})
    lim = resources.get("limits",   {"cpu": "2",    "memory": "8Gi"})

    if hardware == "gpu":
        lim["nvidia.com/gpu"] = resources.get("limits", {}).get("nvidia.com/gpu", "1")

    image = "ollama/ollama:latest" if backend == "ollama" else "vllm/vllm-openai:latest"

    container = client.V1Container(
        name=model_name,
        image=image,
        # Start Ollama server, wait for it, pull the model, then keep running
        command=["sh", "-c", f"ollama serve & sleep 5 && ollama pull {model_id} && wait"],
        ports=[client.V1ContainerPort(container_port=11434, name="api")],
        env=[
            client.V1EnvVar(name="OLLAMA_MODEL", value=model_id),
        ],
        resources=client.V1ResourceRequirements(requests=req, limits=lim),
        readiness_probe=client.V1Probe(
            _exec=client.V1ExecAction(
                command=["sh", "-c", f"ollama list | grep -q {model_id}"],
            ),
            initial_delay_seconds=10,
            period_seconds=10,
            failure_threshold=30,
        ),
    )

    affinity = client.V1Affinity(
        node_affinity=client.V1NodeAffinity(
            preferred_during_scheduling_ignored_during_execution=[
                client.V1PreferredSchedulingTerm(
                    weight=100,
                    preference=client.V1NodeSelectorTerm(
                        match_expressions=[
                            client.V1NodeSelectorRequirement(
                                key="node-type",
                                operator="In",
                                values=[hardware],
                            )
                        ]
                    ),
                )
            ]
        )
    )

    return client.V1Deployment(
        metadata=client.V1ObjectMeta(
            name=f"peag-{model_name}",
            namespace=NAMESPACE,
            labels={
                "app": f"peag-{model_name}",
                "managed-by": "peag",
                "model": model_name,
            },
        ),
        spec=client.V1DeploymentSpec(
            replicas=1,
            selector=client.V1LabelSelector(
                match_labels={"app": f"peag-{model_name}"}
            ),
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    labels={"app": f"peag-{model_name}", "model": model_name}
                ),
                spec=client.V1PodSpec(
                    containers=[container],
                    affinity=affinity,
                ),
            ),
        ),
    )


def _build_service(model_name: str) -> client.V1Service:
    """Build a ClusterIP Service so the gateway can route to this model."""
    return client.V1Service(
        metadata=client.V1ObjectMeta(
            name=f"peag-{model_name}",
            namespace=NAMESPACE,
            labels={
                "app": f"peag-{model_name}",
                "managed-by": "peag",
                "model": model_name,
            },
        ),
        spec=client.V1ServiceSpec(
            selector={"app": f"peag-{model_name}"},
            ports=[client.V1ServicePort(port=11434, target_port=11434, name="api")],
            type="ClusterIP",
        ),
    )


async def spin_up(model_name: str) -> bool:
    """
    Spin up a model deployment and service.
    Returns True if spin-up was triggered, False if already running or no ModelConfig.
    """
    spec = _get_model_config(model_name)
    if not spec:
        logger.error(f"Cannot spin up {model_name} — no ModelConfig found")
        return False

    try:
        _load_kube_config()
        apps_v1 = client.AppsV1Api()
        v1 = client.CoreV1Api()

        # Check if deployment already exists
        try:
            apps_v1.read_namespaced_deployment(
                name=f"peag-{model_name}", namespace=NAMESPACE
            )
            logger.info(f"Deployment for {model_name} already exists")
            return True
        except ApiException as e:
            if e.status != 404:
                raise

        # Create deployment
        deployment = _build_deployment(model_name, spec)
        apps_v1.create_namespaced_deployment(namespace=NAMESPACE, body=deployment)
        logger.info(f"Deployment created for {model_name} — state: warming")

        # Create service so gateway can route to this model
        service = _build_service(model_name)
        try:
            v1.create_namespaced_service(namespace=NAMESPACE, body=service)
            logger.info(f"Service created for {model_name}")
        except ApiException as e:
            if e.status != 409:  # 409 = already exists, fine
                logger.warning(f"Service creation failed for {model_name}: {e}")

        set_state(model_name, ModelState.WARMING)
        return True

    except Exception as e:
        logger.error(f"spin_up failed for {model_name}: {e}")
        set_state(model_name, ModelState.COLD)
        return False


async def scale_to_zero(model_name: str) -> None:
    """Delete the model deployment and service — scales to absolute zero."""
    try:
        _load_kube_config()
        apps_v1 = client.AppsV1Api()
        v1 = client.CoreV1Api()

        # Delete deployment
        apps_v1.delete_namespaced_deployment(
            name=f"peag-{model_name}",
            namespace=NAMESPACE,
            body=client.V1DeleteOptions(propagation_policy="Foreground"),
        )
        logger.info(f"Deployment deleted for {model_name} — state: cold")

        # Delete service
        try:
            v1.delete_namespaced_service(
                name=f"peag-{model_name}", namespace=NAMESPACE
            )
            logger.info(f"Service deleted for {model_name}")
        except ApiException as e:
            if e.status != 404:
                logger.warning(f"Service deletion failed for {model_name}: {e}")

        set_state(model_name, ModelState.COLD)

    except ApiException as e:
        if e.status == 404:
            logger.info(f"Deployment for {model_name} already gone")
            set_state(model_name, ModelState.COLD)
        else:
            logger.error(f"scale_to_zero failed for {model_name}: {e}")


async def wait_until_warm(model_name: str, timeout: int = 900) -> bool:
    """
    Poll until the model deployment is Ready, or timeout.
    Returns True if warm, False if timed out.
    """
    _load_kube_config()
    apps_v1 = client.AppsV1Api()

    for _ in range(timeout // 5):
        await asyncio.sleep(5)
        try:
            dep = apps_v1.read_namespaced_deployment(
                name=f"peag-{model_name}", namespace=NAMESPACE
            )
            ready = dep.status.ready_replicas or 0
            if ready >= 1:
                set_state(model_name, ModelState.WARM)
                logger.info(f"{model_name} is warm and ready")
                return True
        except ApiException:
            pass

    logger.warning(f"{model_name} did not become ready within {timeout}s")
    return False