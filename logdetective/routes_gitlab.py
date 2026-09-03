from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Header,
    Request,
)
from fastapi.responses import Response as BasicResponse

from logdetective.config import SERVER_CONFIG, LOG
from logdetective.database.models import Forge
from logdetective.gitlab import process_gitlab_job_event
from logdetective.models import JobHook

gitlab_router = APIRouter(prefix="/webhook/gitlab")


def is_valid_webhook_secret(forge, x_gitlab_token):
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
    background_tasks: BackgroundTasks,
    request: Request,
    x_gitlab_instance: Annotated[str | None, Header()],
    x_gitlab_token: Annotated[str | None, Header()] = None,
):
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

    # Handle the message in the background so we can return 204 immediately
    gitlab_cfg = SERVER_CONFIG.gitlab.instances[forge.value]
    gitlab_connection = request.app.state.connection_manager.gitlab_connections[
        forge.value
    ]
    gitlab_http_session = request.app.state.connection_manager.gitlab_http_sessions[
        forge.value
    ]
    background_tasks.add_task(
        process_gitlab_job_event,
        gitlab_cfg,
        gitlab_connection,
        gitlab_http_session,
        forge,
        job_hook,
        request.app.state.chat_model,
    )

    # No return value or body is required for a webhook.
    # 204: No Content
    return BasicResponse(status_code=204)
