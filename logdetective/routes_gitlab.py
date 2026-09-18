from typing import Annotated
from uuid import uuid4

from fastapi import (
    APIRouter,
    Header,
    Request,
)
from fastapi.responses import Response as BasicResponse

from logdetective.config import SERVER_CONFIG, LOG
from logdetective.database.models import Forge, EndpointType
from logdetective.database.models.exceptions import TaskConflictError
from logdetective.database.models.tasks import TaskAnalysis, TaskType
from logdetective.models import JobHook
from logdetective.tasks import analyze_gitlab

gitlab_router = APIRouter(prefix="/webhook/gitlab")


def is_valid_webhook_secret(forge: Forge, x_gitlab_token: str | None) -> bool:
    """Check whether the provided x_gitlab_token matches the webhook secret
    specified in the configuration"""

    gitlab_cfg = SERVER_CONFIG.gitlab.instances[forge.value]

    if not gitlab_cfg.webhook_secrets:
        # No secrets specified, so don't bother validating.
        # This is mostly to be used for development.
        return True

    if x_gitlab_token in gitlab_cfg.webhook_secrets:
        return True

    return False


@gitlab_router.post("/job_events")
async def receive_gitlab_job_event_webhook(
    job_hook: JobHook,
    request: Request,
    x_gitlab_instance: Annotated[str | None, Header()],
    x_gitlab_token: Annotated[str | None, Header()] = None,
) -> BasicResponse:
    """Webhook endpoint for receiving job_events notifications from GitLab
    https://docs.gitlab.com/user/project/integrations/webhook_events/#job-events
    lists the full specification for the messages sent for job events."""

    try:
        forge = Forge(x_gitlab_instance)
    except ValueError:
        LOG.critical("%s is not a recognized forge. Ignoring.", x_gitlab_instance)
        return BasicResponse(status_code=400)

    if forge.value not in SERVER_CONFIG.gitlab.instances:
        LOG.warning("%s is a recognized forge but is not configured. Ignoring.", forge.value)
        return BasicResponse(status_code=404)

    if not is_valid_webhook_secret(forge, x_gitlab_token):
        # This request could not be validated, so return a 401
        # (Unauthorized) error.
        return BasicResponse(status_code=401)

    task_id = uuid4()
    payload = {
        "forge": forge.value,
        "job_hook": job_hook.model_dump(mode="json"),
        "api_token_name": request.state.api_token_name,
    }
    try:
        await TaskAnalysis.admit(
            task_id=task_id,
            owner_token_name=request.state.api_token_name,
            task_type=TaskType.GITLAB,
            input_payload=payload,
            request_size=0,
            deferrable_task=analyze_gitlab,
            endpoint=EndpointType.ANALYZE_GITLAB_JOB,
            source_id=f"{forge.value}:{job_hook.build_id}",
        )
    except TaskConflictError:
        return BasicResponse(status_code=409)

    # No return value or body is required for a webhook.
    # 204: No Content
    return BasicResponse(status_code=204)
