import asyncio
import logging
from datetime import datetime, timezone
from croniter import croniter
from kubernetes import client, config as kube_config
from kubernetes.client.rest import ApiException

logger = logging.getLogger(__name__)

CHECK_INTERVAL_SECONDS = 60  # check every minute


def _load_kube_config():
    try:
        kube_config.load_incluster_config()
    except Exception:
        kube_config.load_kube_config()


def _get_all_agent_configs() -> list[dict]:
    try:
        _load_kube_config()
        custom = client.CustomObjectsApi()
        result = custom.list_namespaced_custom_object(
            group="peag.io",
            version="v1alpha1",
            namespace="ai",
            plural="agentconfigs",
        )
        return result.get("items", [])
    except ApiException as e:
        logger.error(f"Failed to list AgentConfigs: {e}")
        return []


def _update_agent_status(agent_name: str, status: str, output: str) -> None:
    """Write run status back to the AgentConfig status field."""
    try:
        _load_kube_config()
        custom = client.CustomObjectsApi()
        now = datetime.now(timezone.utc).isoformat()
        custom.patch_namespaced_custom_object_status(
            group="peag.io",
            version="v1alpha1",
            namespace="ai",
            plural="agentconfigs",
            name=agent_name,
            body={
                "status": {
                    "lastRun": now,
                    "lastRunStatus": status,
                    "lastRunOutput": output[:500],  # truncate for CRD storage
                }
            }
        )
    except Exception as e:
        logger.warning(f"Could not update agent status for {agent_name}: {e}")


def _should_run_now(cron_expression: str) -> bool:
    """Check if the cron expression matches the current minute."""
    try:
        now = datetime.now(timezone.utc)
        cron = croniter(cron_expression, now)
        prev = cron.get_prev(datetime)
        # Fire if the previous scheduled time was within the last 60 seconds
        diff = (now - prev).total_seconds()
        return diff < CHECK_INTERVAL_SECONDS
    except Exception as e:
        logger.error(f"Invalid cron expression '{cron_expression}': {e}")
        return False


async def _run_scheduled_agent(agent_name: str, spec: dict) -> None:
    """Run a scheduled agent and handle the output."""
    from agent_executor import run_agent

    scheduled_prompt = spec.get("scheduledPrompt", "")
    output_channel = spec.get("outputChannel", {})
    agent_display_name = spec.get("name", agent_name)

    if not scheduled_prompt:
        logger.warning(f"Agent {agent_name} has a schedule but no scheduledPrompt")
        return

    logger.info(f"Running scheduled agent: {agent_name}")

    try:
        result = await run_agent(agent_name, scheduled_prompt)
        response = result.get("response", "")
        status = "success"
        logger.info(f"Scheduled agent {agent_name} completed successfully")

        # Send output to configured channel
        channel_type = output_channel.get("type", "none")
        target = output_channel.get("target", "")

        if channel_type == "slack" and target:
            await _send_to_slack(target, agent_display_name, response)
        elif channel_type == "email" and target:
            await _send_email(target, agent_display_name, response)
        elif channel_type == "webhook" and target:
            await _send_webhook(target, agent_display_name, response)

        _update_agent_status(agent_name, status, response)

    except Exception as e:
        logger.error(f"Scheduled agent {agent_name} failed: {e}")
        _update_agent_status(agent_name, "failed", str(e))


async def _send_to_slack(channel: str, agent_name: str, message: str) -> None:
    """Send agent output to a Slack channel via webhook."""
    import httpx
    import os

    webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")
    if not webhook_url:
        logger.warning("SLACK_WEBHOOK_URL not set — cannot send to Slack")
        return

    payload = {
        "channel": channel,
        "text": f"*{agent_name}*\n{message}",
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as http:
            await http.post(webhook_url, json=payload)
        logger.info(f"Sent to Slack channel {channel}")
    except Exception as e:
        logger.error(f"Slack send failed: {e}")


async def _send_email(to: str, agent_name: str, message: str) -> None:
    """Send agent output via email using SMTP."""
    import smtplib
    import os
    from email.mime.text import MIMEText

    smtp_host = os.getenv("SMTP_HOST", "")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_pass = os.getenv("SMTP_PASS", "")

    if not smtp_host:
        logger.warning("SMTP_HOST not set — cannot send email")
        return

    try:
        msg = MIMEText(message)
        msg["Subject"] = f"PEAG Agent Report: {agent_name}"
        msg["From"] = smtp_user
        msg["To"] = to

        def _send_sync():
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)

        await asyncio.to_thread(_send_sync)
        logger.info(f"Email sent to {to}")
    except Exception as e:
        logger.error(f"Email send failed: {e}")


async def _send_webhook(url: str, agent_name: str, message: str) -> None:
    """Send agent output to a generic webhook."""
    import httpx
    payload = {"agent": agent_name, "output": message}
    try:
        async with httpx.AsyncClient(timeout=10.0) as http:
            await http.post(url, json=payload)
        logger.info(f"Webhook sent to {url}")
    except Exception as e:
        logger.error(f"Webhook send failed: {e}")


async def run_agent_scheduler() -> None:
    """Main scheduler loop — checks all agents every minute."""
    logger.info("PEAG agent scheduler started")

    while True:
        try:
            configs = _get_all_agent_configs()

            for ac in configs:
                spec = ac.get("spec", {})
                agent_name = ac["metadata"]["name"]
                schedule = spec.get("schedule", "")

                if not schedule:
                    continue

                if _should_run_now(schedule):
                    logger.info(f"Cron trigger: {agent_name} (schedule: {schedule})")
                    asyncio.create_task(
                        _run_scheduled_agent(agent_name, spec)
                    )

        except Exception as e:
            logger.error(f"Agent scheduler loop error: {e}")

        await asyncio.sleep(CHECK_INTERVAL_SECONDS)
