import datetime
import io
import random
from contextlib import contextmanager
from typing import Generator, Optional
import zipfile

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, session
from flexmock import flexmock
from pytest_mock import MockerFixture

from openai.types.chat.chat_completion import Choice, ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.resources.chat.completions import AsyncCompletions

import koji

from logdetective.server.models import (
    Response,
    Explanation,
)
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
from logdetective.server.compressors import LLMResponseCompressor, RemoteLogCompressor
from logdetective.server.models import GitLabInstanceConfig


MOCK_LOG = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
"""

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


class DatabaseFactory:  # pylint: disable=too-few-public-methods
    @staticmethod
    def get_pg_test_url() -> str:
        """Create PostgreSql connection string to a testing db.
        Database container is started by `tox -e pytest` command,
        connection details for the container are specified in tox.ini"""

        return "postgresql+psycopg2://user:password@localhost:5432/test_db"

    def __init__(self):
        """Connect to a postgres container for testing purposes."""
        self.engine = create_engine(
            self.get_pg_test_url(), connect_args={"connect_timeout": 10}
        )
        self.SessionFactory = sessionmaker(autoflush=True, bind=self.engine)
        flexmock(base, engine=self.engine, SessionFactory=self.SessionFactory)

    @contextmanager
    def make_new_db(self):
        try:
            init()
            yield self.SessionFactory
        finally:
            destroy()


class PopulateDatabase:  # pylint: disable=too-few-public-methods
    def __init__(self):
        self.db_factory = DatabaseFactory()

    @contextmanager
    def populate_db_at_regular_intervals(  # pylint: disable=too-many-positional-arguments
        self,
        interval: datetime.timedelta = datetime.timedelta(minutes=15),
        duration: datetime.timedelta = datetime.timedelta(hours=23),
        end_time: Optional[datetime.datetime] = None,
        endpoint_type: Optional[EndpointType] = EndpointType.ANALYZE,
    ) -> Generator:
        with self.db_factory.make_new_db() as session_factory:  # pylint: disable=contextmanager-generator-missing-cleanup
            end_time = end_time or datetime.datetime.now(datetime.timezone.utc)
            start_time = end_time - duration

            current_time = start_time
            increasing_response_time = 1
            increasing_response_length = 500
            while current_time < end_time:
                id_ = AnalyzeRequestMetrics.create(
                    endpoint=endpoint_type,
                    compressed_log=RemoteLogCompressor.zip_text(
                        "Some log for a failed build"
                    ),
                    request_received_at=current_time,
                )
                response_time = current_time + datetime.timedelta(
                    seconds=increasing_response_time
                )
                increasing_response_length += 500
                AnalyzeRequestMetrics.update(
                    id_=id_,
                    response_sent_at=response_time,
                    response_length=increasing_response_length,
                    response_certainty=70,
                    compressed_response=LLMResponseCompressor(
                        Response(
                            explanation=Explanation(text="a small error", logprobs=[]),
                            response_certainty=0.1,
                        )
                    ).zip_response(),
                )
                current_time += interval
                increasing_response_time += 1
                increasing_response_time = increasing_response_time % 4
                increasing_response_length = increasing_response_length % 5000

            yield session_factory

    @contextmanager
    def populate_db_with_emojis_at_regular_intervals(  # pylint: disable=too-many-positional-arguments
        self,
        interval: datetime.timedelta = datetime.timedelta(minutes=15),
        duration: datetime.timedelta = datetime.timedelta(hours=23),
        end_time: Optional[datetime.datetime] = None,
    ) -> Generator:
        with self.db_factory.make_new_db() as session_factory:  # pylint: disable=contextmanager-generator-missing-cleanup
            end_time = end_time or datetime.datetime.now(datetime.timezone.utc)
            start_time = end_time - duration

            current_time = start_time
            project_id = 333
            job_id = 22
            mr_iid = 1
            comment_id = "a"
            while current_time < end_time:
                GitlabMergeRequestJobs.create(
                    forge=Forge.gitlab_com,
                    project_id=project_id,
                    mr_iid=mr_iid,
                    job_id=job_id,
                )
                id_ = Comments.create(
                    forge=Forge.gitlab_com,
                    project_id=project_id,
                    mr_iid=mr_iid,
                    job_id=job_id,
                    comment_id=comment_id,
                )
                with session_factory() as db_session:
                    comment = Comments.get_by_id(id_)
                    comment.created_at = current_time
                    db_session.add(comment)
                    db_session.flush()
                    db_session.commit()

                Reactions.create_or_update(
                    forge=Forge.gitlab_com,
                    project_id=project_id,
                    mr_iid=mr_iid,
                    job_id=job_id,
                    comment_id=comment_id,
                    reaction_type=random.choice(
                        ["thumbsup", "thumbsdown", "hearth", "confused", "laughing"]
                    ),
                    count=random.randint(1, 10),
                )
                Reactions.create_or_update(
                    forge=Forge.gitlab_com,
                    project_id=project_id,
                    mr_iid=mr_iid,
                    job_id=job_id,
                    comment_id=comment_id,
                    reaction_type=random.choice(
                        ["thumbsup", "thumbsdown", "hearth", "confused", "laughing"]
                    ),
                    count=random.randint(1, 10),
                )

                current_time += interval
                project_id += 100
                job_id += 10
                mr_iid += 1
                if len(comment_id) < 49:
                    comment_id += comment_id[0]
                else:
                    comment_id = chr(ord(comment_id[0]) + 1)

            yield session_factory

    @classmethod
    @contextmanager
    def populate_db(cls, duration=datetime.timedelta, endpoint=EndpointType):
        """Populate the db, one request every 15 minutes.
        and responses increasing for 1 hour, and then back to 1.
        For the last duration time.
        """
        with cls().populate_db_at_regular_intervals(
            duration=duration,
            endpoint_type=endpoint,
        ) as session_factory:
            yield session_factory

    @classmethod
    @contextmanager
    def populate_db_with_emojis(cls, duration=datetime.timedelta):
        """Populate the db, one comment every 15 minutes."""
        with cls().populate_db_with_emojis_at_regular_intervals(
            duration=duration,
        ) as session_factory:
            yield session_factory


@pytest.fixture(scope="function")
@contextmanager
def populate_db_with_analyze_request_every_15_minutes_for_15_hours():
    duration = datetime.timedelta(hours=15)
    with PopulateDatabase().populate_db_at_regular_intervals(
        duration=duration
    ) as session_factory:
        yield session_factory


@pytest.fixture(scope="function")
@contextmanager
def populate_db_with_analyze_request_every_15_minutes_for_9_days():
    duration = datetime.timedelta(days=9)
    with PopulateDatabase().populate_db_at_regular_intervals(
        duration=duration
    ) as session_factory:
        yield session_factory


@pytest.fixture(scope="function")
@contextmanager
def populate_db_with_analyze_request_every_15_minutes_for_3_weeks():
    duration = datetime.timedelta(weeks=3)
    with PopulateDatabase().populate_db_at_regular_intervals(
        duration=duration
    ) as session_factory:
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
def build_log():
    return {"build_log": flexmock(url="https://example.com/logs/123")}


@pytest.fixture
def mock_AnalyzeRequestMetrics():
    update_response_metrics = flexmock()
    all_metrics = (
        flexmock().should_receive("first").and_return(update_response_metrics).mock()
    )
    query = flexmock().should_receive("filter_by").and_return(all_metrics).mock()
    flexmock(session.Session).should_receive("query").and_return(query)
    flexmock(session.Session).should_receive("add").and_return()
    flexmock(AnalyzeRequestMetrics).should_receive("create").once().and_return(1)


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
        data={
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


def create_mock_koji_session(mocker, task_id, method):
    mock_session = mocker.Mock()
    mock_session.getTaskInfo.return_value = {
        "id": task_id,
        "state": koji.TASK_STATES["FAILED"],
        "method": method,
        "arch": "x86_64",
    }

    mock_session.getTaskResult.return_value = {
        "faultString": "BuildError: error building package (arch x86_64), mock exited with status 1; see build.log or root.log for more information"  # pylint: disable=line-too-long
    }

    # Mock the build log response
    mock_session.listTaskOutput.return_value = {
        "build.log": {
            "st_size": "43",
        },
    }

    mock_session.downloadTaskOutput.return_value = (
        b"Error: Build failed\nDetailed error message"
    )
    return mock_session
