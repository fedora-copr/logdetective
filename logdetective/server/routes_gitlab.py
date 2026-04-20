from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Header,
    Request,
)
from fastapi.responses import Response as BasicResponse
from gitlab import Gitlab

from logdetective.server.config import SERVER_CONFIG, LOG
from logdetective.server.database.models import Forge
from logdetective.server.emoji import collect_emojis_for_mr
from logdetective.server.gitlab import process_gitlab_job_event
from logdetective.server.models import (
    EmojiHook,
    JobHook,
)

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
        request.app.state.openai_chat_model,
    )

    # No return value or body is required for a webhook.
    # 204: No Content
    return BasicResponse(status_code=204)


# A lookup table for whether we are currently processing a given merge request
# The key is the tuple (Forge, ProjectID, MRID) and the value is a boolean
# indicating whether we need to re-trigger the lookup immediately after
# completion due to another request coming in during processing.
# For example: {("https://gitlab.example.com", 23, 2): False}
emoji_lookup = {}


@gitlab_router.post("/emoji_events")
async def receive_gitlab_emoji_event_webhook(
    x_gitlab_instance: Annotated[str | None, Header()],
    x_gitlab_token: Annotated[str | None, Header()],
    emoji_hook: EmojiHook,
    background_tasks: BackgroundTasks,
    request: Request,
):
    """Webhook endpoint for receiving emoji event notifications from Gitlab
    https://docs.gitlab.com/user/project/integrations/webhook_events/#emoji-events
    lists the full specification for the messages sent for emoji events"""

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

    if not emoji_hook.merge_request:
        # This is not a merge request event. It is probably an emoji applied
        # to some other "awardable" entity. Just ignore it and return.
        LOG.debug("Emoji event is not related to a merge request. Ignoring.")
        return BasicResponse(status_code=204)

    # We will re-process all the emojis on this merge request, to ensure that
    # we haven't missed any messages, since webhooks do not provide delivery
    # guarantees.

    # Check whether this request is already in progress.
    # We are single-threaded, so we can guarantee that the table won't change
    # between here and when we schedule the lookup.
    key = (
        forge,
        emoji_hook.merge_request.target_project_id,
        emoji_hook.merge_request.iid,
    )
    if key in emoji_lookup:
        # It's already in progress, so we do not want to start another pass
        # concurrently. We'll set the value to True to indicate that we should
        # re-enqueue this lookup after the currently-running one concludes. It
        # is always safe to set this to True, even if it's already True. If
        # multiple requests come in during processing, we only need to re-run
        # it a single time, since it will pick up all the ongoing changes. The
        # worst-case situation is the one where we receive new requests just
        # after processing starts, which will cause the cycle to repeat again.
        # This should be very infrequent, as emoji events are computationally
        # rare and very quick to process.
        emoji_lookup[key] = True
        LOG.info("MR Emojis already being processed for %s. Rescheduling.", key)
        return BasicResponse(status_code=204)

    # Inform the lookup table that we are processing this emoji
    emoji_lookup[key] = False

    gitlab_connection = request.app.state.connection_manager.gitlab_connections[
        forge.value
    ]
    # Create a background task to process the emojis on this Merge Request.
    background_tasks.add_task(
        schedule_emoji_collection_for_mr,
        forge,
        gitlab_connection,
        emoji_hook.merge_request.target_project_id,
        emoji_hook.merge_request.iid,
        background_tasks,
    )

    # No return value or body is required for a webhook.
    # 204: No Content
    return BasicResponse(status_code=204)


async def schedule_emoji_collection_for_mr(
    forge: Forge,
    gitlab_connection: Gitlab,
    project_id: int,
    mr_iid: int,
    background_tasks: BackgroundTasks,
):
    """Background task to update the database on emoji reactions"""

    key = (forge, project_id, mr_iid)

    # FIXME: Look up the connection from the Forge  # pylint: disable=fixme
    # gitlab_conn = SERVER_CONFIG.gitlab.instances[forge.value].get_connection()

    LOG.debug("Looking up emojis for %s, %d, %d", forge, project_id, mr_iid)
    await collect_emojis_for_mr(project_id, mr_iid, gitlab_connection)

    # Check whether we've been asked to re-schedule this lookup because
    # another request came in while it was processing.
    if emoji_lookup[key]:
        # The value is Truthy, which tells us to re-schedule
        # Reset the boolean value to indicate that we're underway again.
        emoji_lookup[key] = False
        background_tasks.add_task(
            schedule_emoji_collection_for_mr,
            forge,
            gitlab_connection,
            project_id,
            mr_iid,
            background_tasks,
        )
        return

    # We're all done, so clear this entry out of the lookup
    del emoji_lookup[key]
