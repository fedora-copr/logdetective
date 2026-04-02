import datetime
import io
from itertools import cycle, count
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator
import zipfile
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from flexmock import flexmock
from pytest_mock import MockerFixture

from openai.types.chat.chat_completion import Choice, ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.resources.chat.completions import AsyncCompletions

import koji

from logdetective.server.models import (
    ArtifactFile,
    Response,
    Explanation,
    Config,
)
from logdetective.server import gitlab
from logdetective.server.database import base
from logdetective.server.database.base import init, destroy
from logdetective.server.database.models import (
    AnalyzeRequestMetrics,
    EndpointType,
    GitlabMergeRequestJobs,
    Comments,
    Reactions,
    Forge,
)
from logdetective.server.compressors import LLMResponseCompressor
from logdetective.server.models import GitLabInstanceConfig


MOCK_LOG = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
"""

MOCK_EXPLANATION = "Mock explanation"

ARCHES = [
    "x86_64",
    "aarch64",
    "ppc64le",
    "riscv",
    "s390x",
]

SIMPLE_METHODS = ["buildArch", "buildSRPMFromSCM"]

SUBTASK_ARCHES = [
    (133801422, "x86_64"),
    (133801421, "aarch64"),
    (133801420, "ppc64le"),
]

EXAMPLE_TASK_ID = 133858346


# Purpose of these tests is not testing the compression which takes substantial time.
# So we use a reference to a precomputed call to populate the DB.
_PRECOMPUTED_COMPRESSED_RESPONSE = LLMResponseCompressor(
    Response(
        explanation=Explanation(text="a small error", logprobs=[]),
    )
).zip_response()


class DatabaseFactory:  # pylint: disable=too-few-public-methods
    @staticmethod
    def get_pg_test_url() -> str:
        """Create PostgreSql connection string to a testing db.
        Database container is started by `tox -e pytest` command,
        connection details for the container are specified in tox.ini"""

        return "postgresql+asyncpg://user:password@localhost:5432/test_db"

    def __init__(self):
        """Connect to a postgres container for testing purposes."""
        self.engine = create_async_engine(
            self.get_pg_test_url(), connect_args={"command_timeout": 10}, pool_pre_ping=True
        )
        self.SessionFactory = async_sessionmaker(autoflush=True, bind=self.engine)
        flexmock(base, engine=self.engine, SessionFactory=self.SessionFactory)

    @asynccontextmanager
    async def make_new_db(self):
        try:
            await init()
            yield self.SessionFactory
        finally:
            await destroy()


class PopulateDatabase:  # pylint: disable=too-few-public-methods
    def __init__(self):
        self.db_factory = DatabaseFactory()

    @asynccontextmanager
    async def populate_db_at_regular_intervals(  # pylint: disable=too-many-positional-arguments
        self,
        interval: datetime.timedelta = datetime.timedelta(minutes=15),
        duration: datetime.timedelta = datetime.timedelta(hours=23),
        end_time: Optional[datetime.datetime] = None,
        endpoint_type: Optional[EndpointType] = EndpointType.ANALYZE,
    ) -> AsyncGenerator:
        # pylint: disable=contextmanager-generator-missing-cleanup
        async with self.db_factory.make_new_db() as session_factory:
            end_time = end_time or datetime.datetime.now(datetime.timezone.utc)
            start_time = end_time - duration

            response_times = cycle([1, 2, 3, 4])
            response_lengths = cycle([1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 500])
            current_time = start_time
            while current_time < end_time:
                id_ = await AnalyzeRequestMetrics.create(
                    endpoint=endpoint_type,
                    request_received_at=current_time,
                )
                response_time = current_time + datetime.timedelta(
                    seconds=next(response_times)
                )
                await AnalyzeRequestMetrics.update(
                    id_=id_,
                    response_sent_at=response_time,
                    response_length=next(response_lengths),
                    compressed_response=_PRECOMPUTED_COMPRESSED_RESPONSE
                )
                current_time += interval

            yield session_factory

    @classmethod
    @asynccontextmanager
    async def populate_db(cls, duration=datetime.timedelta, endpoint=EndpointType):
        """Populate the db, one request every 15 minutes.
        and responses increasing for 1 hour, and then back to 1.
        For the last duration time.
        """
        async with cls().populate_db_at_regular_intervals(
            duration=duration,
            endpoint_type=endpoint,
        ) as session_factory:
            yield session_factory

    @classmethod
    @asynccontextmanager
    async def populate_db_with_analysis_records(
        cls,
        time_anchor: datetime.datetime,
        records: list[tuple[datetime.timedelta, float]],
        endpoint: EndpointType
    ):
        """
        Populate the DB with request metrics based on a list of metadata

        `time_anchor`: timestamp from which the actual mock API call timestamps are calculated.
        This time is usually aligned to an hour mark or midnight, so that we can have
        more control over which buckets would the request metrics land in.
        `records`: relative timedeltas to `time_anchor` and API response durations
        """
        response_lengths = cycle([500, 1000, 1500, 2000])
        # pylint: disable=contextmanager-generator-missing-cleanup
        async with cls().db_factory.make_new_db() as session_factory:
            for offset, runtime in records:
                request_timestamp = time_anchor - offset
                id_ = await AnalyzeRequestMetrics.create(endpoint, request_timestamp)
                response_timestamp = request_timestamp + datetime.timedelta(seconds=runtime)
                await AnalyzeRequestMetrics.update(
                    id_,
                    response_timestamp,
                    next(response_lengths),
                    _PRECOMPUTED_COMPRESSED_RESPONSE,
                )
            yield session_factory

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    @staticmethod
    async def _cascade_db_populate_for_emojis(
        session_context: async_sessionmaker[AsyncSession],
        project_id: int,
        mr_iid: int,
        job_id: int,
        comment_id: int,
        timestamp: datetime.datetime,
        emoji_data: dict[str, int],
    ):
        """
        A helper function that, given the necessary mock IDs,
        inserts all necessary DB entries along with `emoji_data` in the DB.
        """
        await GitlabMergeRequestJobs.create(
            Forge.gitlab_com, project_id, mr_iid, job_id,
        )
        id_ = await Comments.create(
            Forge.gitlab_com, project_id, mr_iid, job_id, comment_id,
        )
        async with session_context() as db_session:
            comment = await Comments.get_by_id(id_)
            comment.created_at = timestamp
            db_session.add(comment)
            await db_session.flush()
            await db_session.commit()
        for emoji_type, emoji_count in emoji_data.items():
            await Reactions.create_or_update(
                Forge.gitlab_com, project_id, mr_iid, job_id, comment_id, emoji_type, emoji_count,
            )

    @classmethod
    @asynccontextmanager
    async def populate_db_with_emoji_records(
        cls,
        time_anchor: datetime.datetime,
        emoji_records: list[tuple[datetime.timedelta, dict[str, int]]]
    ):
        """
        Populate the database with emojis based on the list of metadata.

        `emoji_records`: one record has a timedelta relative to `time_anchor`, see
        `populate_db_with_analysis_records()` and emoji -> emoji count map.
        """
        projects = count(start=3, step=3)
        jobs = count(start=2, step=2)
        merge_requests = count(start=1)
        comments = cycle("abcdefghijklmnopqrstuvwxyz")

        # pylint: disable=contextmanager-generator-missing-cleanup
        db = cls()
        async with db.db_factory.make_new_db() as session_factory:
            for offset, emoji_data in emoji_records:
                await db._cascade_db_populate_for_emojis(
                    session_factory,
                    next(projects),
                    next(merge_requests),
                    next(jobs),
                    next(comments),
                    time_anchor - offset,
                    emoji_data
                )
            yield session_factory


@pytest.fixture()
def mock_chat_completions(monkeypatch, request):
    """Returns mock ChatCompletion response asynchronously."""
    mock_message = request.param

    async def mock_create(*args, **kwargs):
        completion = ChatCompletion(
            id="mock_completion",
            created=0,
            object="chat.completion",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        content=mock_message,
                        role="assistant",
                    ),
                )
            ],
            model="mock_completion_model",
        )

        return completion

    monkeypatch.setattr(AsyncCompletions, "create", mock_create)


@pytest.fixture
def build_log_request(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def build_log_url():
    return {"payload": flexmock(url="https://example.com/logs/123", files=None)}


@pytest.fixture
def build_log_two_files():
    return {
        "payload": flexmock(
            url=None,
            files=[
                ArtifactFile(name="builder-live.log", content=MOCK_LOG),
                ArtifactFile(name="backend.log", content=MOCK_LOG)
            ]
        )
    }


@pytest.fixture
def build_log_one_file():
    return {
        "payload": flexmock(
            url=None,
            files=[
                ArtifactFile(name="build.log", content=MOCK_LOG)
            ]
        )
    }


@pytest.fixture
def mock_AnalyzeRequestMetrics(mocker: MockerFixture):
    mock_create = mocker.patch(
        "logdetective.server.database.models.AnalyzeRequestMetrics.create",
        new_callable=AsyncMock,
        return_value=1,
    )

    mock_update = mocker.patch(
        "logdetective.server.database.models.AnalyzeRequestMetrics.update",
        new_callable=AsyncMock,
    )

    return {"mock_create": mock_create, "mock_update": mock_update}


class MockGitlabJob:
    # pylint: disable=too-few-public-methods
    """A mock representation of a gitlab.v4.objects.ProjectJob object."""

    def __init__(self, project_id: int, job_id: int):
        self.project_id = project_id
        self.id = job_id
        self.job_artifacts_called = False

    def artifacts(self, streamed: bool = False, action=None):
        """Mock method for downloading artifacts."""
        self.job_artifacts_called = True
        # In the real function, this method's logic is handled by the mock
        # for asyncio.to_thread, so this body can be empty.


def create_zip_archive(files_to_add: dict[str, str]) -> bytes:
    """Creates an in-memory zip archive from a dictionary of filename: content."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for filename, content in files_to_add.items():
            zip_file.writestr(filename, content.encode("utf-8"))
    zip_buffer.seek(0)
    return zip_buffer.read()


def mock_artifact_download(mocker: MockerFixture, zip_content: bytes):
    """Mocks the asyncio.to_thread call that simulates the artifact download."""

    async def dummy_coro():
        pass

    def mock_side_effect(func, *args, **kwargs):
        action = kwargs.get("action")
        if action and callable(action):
            action(zip_content)
        return dummy_coro()

    mocker.patch("asyncio.to_thread", side_effect=mock_side_effect)


@pytest.fixture
def gitlab_cfg() -> GitLabInstanceConfig:
    """Provides a standard GitLabInstanceConfig for tests."""
    return GitLabInstanceConfig(
        name="mocked_gitlab",
        **{
            "url": "https://gitlab.com",
            "api_path": "/api/v4",
            "api_token": "empty",
            "max_artifact_size": 300,
        },
    )


@pytest.fixture
def mock_job() -> MockGitlabJob:
    """Provides a standard MockGitlabJob for tests."""
    return MockGitlabJob(project_id=42, job_id=101)


def create_mock_koji_session(
    mocker, task_id, method, arch="x86_64", list_task_output=True
):
    """Mock koji session. Returns responses to `getTaskOutput`, `listTaskOutput`
    and `downloadTaskOutput` methods. If `list_task_output` is set to `False`
    will instead return `None` for the `listTaskOutput`."""
    mock_session = mocker.Mock()

    mock_session.getTaskInfo.return_value = {
        "id": task_id,
        "state": koji.TASK_STATES["FAILED"],
        "method": method,
        "arch": arch,
    }

    mock_session.getTaskResult.return_value = {
        "faultString": "BuildError: error building package (arch x86_64), mock exited with status 1; see build.log or root.log for more information"  # pylint: disable=line-too-long
    }

    mock_session.downloadTaskOutput.return_value = (
        b"Error: Build failed\nDetailed error message"
    )

    if list_task_output:
        # Mock the build log response
        mock_session.listTaskOutput.return_value = {
            "build.log": {
                "st_size": "43",
            },
        }
    else:
        mock_session.listTaskOutput.return_value = None

    return mock_session


def create_mock_client_response(mocker, content_length: int | None = None):
    """Creates a mock aiohttp.ClientSession that can be awaited."""
    # This is the mock response object that head() will eventually return.
    mock_response = mocker.Mock()
    mock_response.headers = {}
    if content_length:
        mock_response.headers["content-length"] = str(content_length)

    # This is the mock for the session object itself.
    mock_session = mocker.MagicMock()
    # We configure its `head` method to be an async function (coroutine)
    # that returns our mock response.
    mock_session.head = AsyncMock(return_value=mock_response)
    return mock_session


@pytest_asyncio.fixture
def mock_config():
    server_config = Config(
        **{
            "gitlab": {
                "gitlab.com": {
                    "url": "https://gitlab.com",
                    "api_token": "abc",
                    "max_artifact_size": 1234567,
                }
            },
            "extractor": {"max_clusters": 1},
            "inference": {
                "model": "some.gguf",
                "max_tokens": -1,
                "api_token": "def",
                "temperature": 1,
                "url": "http://inference-server:8000",
            },
            "general": {
                "packages": ["a project", "python3-.*"],
                "excluded_packages": ["python3-excluded", "python3-more-exclusions.*"],
            },
        }
    )
    flexmock(gitlab).should_receive("SERVER_CONFIG").and_return(server_config)

    return {"server_config": server_config}
