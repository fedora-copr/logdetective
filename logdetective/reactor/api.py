import aiohttp
import gitlab
import jinja2

from typing import Annotated
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks, Depends
from fastapi.responses import Response as BasicResponse

from logdetective.reactor.config import get_config
from logdetective.reactor.gitlab import process_gitlab_job_event
from logdetective.reactor.logging import get_log
from logdetective.reactor.models import JobHook
from logdetective.reactor.dependencies import (
    HttpConnections,
    get_http_connections,
    get_jinja_env,
)


REACTOR_CONFIG = get_config()
LOG = get_log()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the jinja2 templates
    # Locate and load the comment template
    script_path = Path(__file__).resolve().parent
    template_path = Path(script_path, "templates")
    app.jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_path))

    # Establish a python-gitlab session
    app.gitlab_conn = gitlab.Gitlab(
        url=REACTOR_CONFIG.gitlab.url, private_token=REACTOR_CONFIG.gitlab.api_token
    )
    # Establish an HTTP connection to the Gitlab API
    # This will be used in places that python-gitlab can't handle, such as
    # getting the size of artifacts.zip
    app.gitlab_http = aiohttp.ClientSession(
        base_url=REACTOR_CONFIG.gitlab.url,
        headers={"Authorization": f"Bearer {REACTOR_CONFIG.gitlab.api_token}"},
    )
    # Establish the Log Detective API connection
    app.logdetective_http = aiohttp.ClientSession(
        base_url=REACTOR_CONFIG.logdetective.url,
        timeout=aiohttp.ClientTimeout(total=5, connect=3.07),
    )

    # The "yield" here ensures that the rest of this function is only called
    # once the application is shutting down. Everything before the "yield" is
    # setup() and everything after is teardown().
    yield

    # Cleanly shut down the HTTP sessions
    await app.logdetective_http.close()
    await app.gitlab_http.close()

    # The gitlab_conn object has no destructor to call.


# Set up the FastAPI application
app = FastAPI(lifespan=lifespan)


@app.post("/webhook/gitlab/job_events")
async def receive_gitlab_job_event_webhook(
    job_hook: JobHook,
    background_tasks: BackgroundTasks,
    jinja_env: Annotated[jinja2.Environment, Depends[get_jinja_env]],
    http_connections: Annotated[HttpConnections, Depends[get_http_connections]],
):
    """Webhook endpoint for receiving job_events notifications from GitLab
    https://docs.gitlab.com/user/project/integrations/webhook_events/#job-events
    lists the full specification for the messages sent for job events."""

    # Handle the message in the background so we can return 200 immediately
    background_tasks.add_task(
        process_gitlab_job_event,
        job_hook,
        jinja_env,
        http_connections,
        REACTOR_CONFIG.general.packages,
    )

    # No return value or body is required for a webhook.
    # 204: No Content
    return BasicResponse(status_code=204)
