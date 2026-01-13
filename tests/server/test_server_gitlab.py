import os
import io
import zipfile
import pytest
import pytest_asyncio
import aiohttp
import responses
import aioresponses
from gitlab import Gitlab
from packaging.version import Version
from pytest_mock import MockerFixture
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from aiolimiter import AsyncLimiter
from fastapi import HTTPException

from flexmock import flexmock

from tests.server.test_helpers import (
    DatabaseFactory,
    mock_chat_completions,
    create_zip_archive,
    mock_artifact_download,
    mock_job,
    gitlab_cfg,
    create_mock_client_response,
)

from logdetective.extractors import DrainExtractor
from logdetective.server.gitlab import (
    is_eligible_package,
    retrieve_and_preprocess_koji_logs,
    check_artifacts_file_size,
)
from logdetective.server.server import process_gitlab_job_event
from logdetective.server.models import JobHook, GitLabInstanceConfig, Config
from logdetective.server.compressors import RemoteLogCompressor
from logdetective.server.database.models import (
    AnalyzeRequestMetrics,
    Forge,
    GitlabMergeRequestJobs,
)
from logdetective.server import gitlab, llm
from logdetective.server.exceptions import LogsTooLargeError


@pytest_asyncio.fixture
def mock_config():
    server_config = Config(
        data={
            "gitlab": {
                "gitlab.com": {
                    "api_url": "https://gitlab.com",
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
                "url": "http://llama-cpp-server:8000",
            },
            "general": {
                "packages": ["a project", "python3-.*"],
                "excluded_packages": ["python3-excluded", "python3-more-exclusions.*"],
            },
        }
    )
    flexmock(gitlab).should_receive("SERVER_CONFIG").and_return(server_config)
    flexmock(llm).should_receive("SERVER_CONFIG").and_return(server_config)


def create_zip_content(filepath) -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(filepath + "task_failed.log", "root.log")
        zip_file.writestr(filepath + "root.log", "ERROR: some sort of error")

    zip_buffer.seek(0)
    return zip_buffer.read()


@pytest_asyncio.fixture
async def mock_job_hook():
    job_hook = JobHook(
        object_kind="build",
        build_id=123,
        pipeline_id=456,
        build_name="build_centos_stream_rpm",
        build_status="failed",
        project_id=678,
    )

    project_content = {"name": "a project", "id": 678, "web_url": "an url"}
    failed_job_content = {
        "commit": {"author_email": "admin@example.com", "author_name": "Administrator"},
        "coverage": None,
        "allow_failure": False,
        "created_at": "2015-12-24T15:51:21.880Z",
        "started_at": "2015-12-24T17:54:30.733Z",
        "finished_at": "2015-12-24T17:54:31.198Z",
        "duration": 0.465,
        "queued_duration": 0.010,
        "artifacts_expire_at": "2016-01-23T17:54:31.198Z",
        "tag_list": ["docker runner", "macos-10.15"],
        "id": 1,
        "name": "rubocop",
        "pipeline": {"id": 456, "project_id": 678},
        "ref": "main",
        "artifacts": [],
        "runner": None,
        "stage": "test",
        "status": "failed",
        "tag": False,
        "web_url": "https://example.com/foo/bar/-/jobs/1",
        "user": {"id": 1},
    }
    pipeline_content = {
        "id": 456,
        "project_id": 678,
        "status": "pending",
        "ref": "refs/merge-requests/99/source",
        "before_sha": "a91957a858320c0e17f3a0eca7cfacbff50ea29a",
        "tag": False,
        "yaml_errors": None,
        "user": {
            "name": "Administrator",
            "username": "root",
            "id": 1,
            "state": "active",
            "avatar_url": (
                "http://www.gravatar.com/avatar/"
                "e64c7d89f26bd1972efa854d13d7dd61?s=80&d=identicon"
            ),
            "web_url": "http://localhost:3000/root",
        },
        "created_at": "2016-08-11T11:28:34.085Z",
        "updated_at": "2016-08-11T11:32:35.169Z",
        "started_at": None,
        "finished_at": "2016-08-11T11:32:35.145Z",
        "committed_at": None,
        "duration": None,
        "queued_duration": 0.010,
        "coverage": None,
        "web_url": "https://example.com/foo/bar/pipelines/46",
        "source": "merge_request_event",
    }

    mr_content = {
        "id": 1,
        "iid": 99,
        "project_id": 678,
        "title": "test",
        "description": "some change in a project",
        "state": "merged",
        "merged_by": {
            "id": 87854,
            "name": "Merger Name",
            "username": "merger name",
            "state": "active",
            "avatar_url": "https://gitlab.example.com/uploads/"
            "-/system/user/avatar/87854/avatar.png",
            "web_url": "https://gitlab.com/Merger",
        },
        "reviewers": [
            {
                "id": 2,
                "name": "Reviewer Name",
                "username": "reviewer name",
                "state": "active",
                "avatar_url": "https://www.gravatar.com/avatar/"
                "956c92487c6f6f7616b536927e22c9a0?s=80&d=identicon",
                "web_url": "http://gitlab.example.com//reviewer",
            }
        ],
    }
    note_content = {}

    mocked_headers = {
        "Content-Length": "12345",
        "Content-Type": "application/zip",
        "ETag": "abc123",
    }

    mocked_llama_response = """
{
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello!"
      },
      "logprobs": {
        "content": [
          {
            "token": "Hello",
            "logprob": -0.31725305,
            "bytes": [72, 101, 108, 108, 111],
            "top_logprobs": [
              {
                "token": "Hello",
                "logprob": -0.31725305,
                "bytes": [72, 101, 108, 108, 111]
              },
              {
                "token": "Hi",
                "logprob": -1.3190403,
                "bytes": [72, 105]
              }
            ]
          },
          {
            "token": "!",
            "logprob": -0.02380986,
            "bytes": [
              33
            ],
            "top_logprobs": [
              {
                "token": "!",
                "logprob": -0.02380986,
                "bytes": [33]
              },
              {
                "token": " there",
                "logprob": -3.787621,
                "bytes": [32, 116, 104, 101, 114, 101]
              }
            ]
          }
        ]
      },
      "finish_reason": "stop"
    }
  ]
}
"""

    with responses.RequestsMock() as sync_rsps:
        with aioresponses.aioresponses() as async_rsps:
            async_rsps.head(
                url="https://gitlab.com/api/v4/projects/678/jobs/1/artifacts",
                status=200,
                headers=mocked_headers,
            )
            async_rsps.post(
                url="http://llama-cpp-server:8000/v1/chat/completions",
                body=mocked_llama_response,
            )
            sync_rsps.add(
                method=responses.GET,
                url="https://gitlab.com/api/v4/projects/678",
                json=project_content,
                content_type="application/json",
                status=200,
            )
            sync_rsps.add(
                method=responses.GET,
                url="https://gitlab.com/api/v4/projects/678/jobs/123",
                json=failed_job_content,
                content_type="application/json",
                status=200,
            )
            sync_rsps.add(
                method=responses.GET,
                url="https://gitlab.com/api/v4/projects/678/pipelines/456",
                json=pipeline_content,
                content_type="application/json",
                status=200,
            )
            sync_rsps.add(
                method=responses.GET,
                url="https://gitlab.com/api/v4/projects/678/jobs/1/artifacts",
                content_type="application/zip",
                body=create_zip_content("kojilogs/noarch-XXXXXX/x86_64-XXXXXX/"),
                status=200,
            )
            sync_rsps.add(
                method=responses.GET,
                url="https://gitlab.com/api/v4/projects/678/merge_requests/99",
                json=mr_content,
                content_type="application/json",
                status=200,
            )
            sync_rsps.add(
                method=responses.POST,
                url="https://gitlab.com/api/v4/projects/678/merge_requests/99/discussions",
                json={
                    "id": 1,
                    "notes": [
                        {
                            "id": 123,
                            "body": "Your comment text here",
                            "author": {
                                "id": 1,
                                "username": "username",
                                "name": "User Name",
                            },
                            "created_at": "2023-01-01T00:00:00Z",
                            "updated_at": "2023-01-01T00:00:00Z",
                        }
                    ],
                },
                status=200,
            )
            sync_rsps.add(
                method=responses.GET,
                url="https://gitlab.com/api/v4/projects/678/merge_requests/99/"
                "discussions/1/notes/123",
                json=note_content,
                content_type="application/json",
                status=200,
            )
            sync_rsps.add(
                method=responses.PUT,
                url="https://gitlab.com/api/v4/projects/678/merge_requests/99/discussions/1/notes",
                json=note_content,
                content_type="application/json",
                status=200,
            )
            yield sync_rsps, async_rsps, job_hook


@pytest.mark.parametrize(
    "mock_chat_completions", ["This is a mock message"], indirect=True
)
@pytest.mark.skipif(
    Version(aioresponses.__version__) < Version("0.7.8"),
    reason="aioresponses lacks support for base_url",
)
@pytest.mark.asyncio
async def test_process_gitlab_job_event(
    mocker,
    mock_config,
    mock_job_hook,
    mock_chat_completions,
    gitlab_cfg,
):
    async with DatabaseFactory().make_new_db() as session_factory:
        _, _, job_hook = mock_job_hook

        http_session = aiohttp.ClientSession()
        async_request_limiter = AsyncLimiter(100)
        gitlab_connection = Gitlab(
            url="https://gitlab.com/",
        )
        mock_http_session = create_mock_client_response(
            mocker, content_length=gitlab_cfg.max_artifact_size
        )
        await process_gitlab_job_event(
            gitlab_cfg=gitlab_cfg,
            gitlab_connection=gitlab_connection,
            http_session=mock_http_session,
            forge=Forge.gitlab_com,
            job_hook=job_hook,
            async_request_limiter=async_request_limiter,
            extractors=[DrainExtractor()],
        )
        async with session_factory() as db_session:
            query = (
                select(AnalyzeRequestMetrics)
                .where(AnalyzeRequestMetrics.id == 1)
                .options(
                    selectinload(AnalyzeRequestMetrics.mr_job).selectinload(
                        GitlabMergeRequestJobs.comment
                    )
                )
            )
            query_results = await db_session.execute(query)
            metric = query_results.scalars().first()
            assert isinstance(metric, AnalyzeRequestMetrics)
            assert (
                RemoteLogCompressor.unzip(metric.compressed_log)  # noqa: W504 flake vs ruff
                == "ERROR: some sort of error"  # noqa: W503 flake vs lint
            )

            assert metric.mr_job.comment
        await http_session.close()


@pytest.mark.asyncio
async def test_is_eligible_package(mock_config):
    # Test an explicit full-text value
    assert is_eligible_package("a project")

    # Test a non-existent package name
    assert not is_eligible_package("a fake project")

    # Test a regular-expression match
    assert is_eligible_package("python3-logdetective")

    # Test an excluded explicit name
    assert not is_eligible_package("python3-excluded")

    # Test an excluded regular expression
    assert not is_eligible_package("python3-more-exclusions-foo")


# Regression test for https://github.com/fedora-copr/logdetective/issues/292
# To test this, comment out the @pytest.mark.skip decorator
@pytest.mark.skip(reason="Requires real network access and a valid token")
@pytest.mark.asyncio
async def test_regression_unknown_arch_logs(mocker: MockerFixture):
    gl_token = os.environ.get("LD_GITLAB_TOKEN", None)

    gitlab_cfg = GitLabInstanceConfig(
        name="Live Network Test Instance",
        data={
            "api_url": "https://gitlab.com",
            "api_token": gl_token,
            "max_artifact_size": 120,
        },
    )
    gitlab_connection = Gitlab(
        url=gitlab_cfg.url,
        private_token=gitlab_cfg.api_token,
        timeout=gitlab_cfg.timeout,
    )

    project = gitlab_connection.projects.get(23665037)
    job = project.jobs.get(10226618670)
    mock_session = mocker.Mock()
    url, open_file = await retrieve_and_preprocess_koji_logs(
        gitlab_cfg=gitlab_cfg, job=job, http_session=mock_session
    )
    open_file.close()

    assert "task_failed.log" not in url
    assert "build.log" in url
    assert url.startswith(gitlab_cfg.url)


@pytest.mark.asyncio
async def test_fallback_to_task_failed_log_if_no_match(
    mocker: MockerFixture, gitlab_cfg, mock_job
):
    """
    Tests that if task_failed.log doesn't mention a specific log,
    it returns the task_failed.log itself.
    """
    log_content = "A generic failure occurred. Check Koji for details."
    files = {"kojilogs/x86_64-build/task_failed.log": log_content}
    zip_content = create_zip_archive(files)
    mock_artifact_download(mocker, zip_content)
    mocker.patch(
        "logdetective.server.gitlab.check_artifacts_file_size", return_value=True
    )

    mock_session = mocker.AsyncMock()

    log_url, log_file = await retrieve_and_preprocess_koji_logs(
        gitlab_cfg=gitlab_cfg, job=mock_job, http_session=mock_session
    )

    assert "artifacts/kojilogs/x86_64-build/task_failed.log" in log_url
    assert log_file.read().decode("utf-8") == log_content
    assert log_url.startswith(gitlab_cfg.url)


@pytest.mark.asyncio
async def test_raises_file_not_found_on_no_failures(
    mocker: MockerFixture, gitlab_cfg, mock_job
):
    """
    Tests that FileNotFoundError is raised if no task_failed.log is found.
    """
    files = {"kojilogs/x86_64-build/build.log": "This build was successful."}
    zip_content = create_zip_archive(files)
    mock_artifact_download(mocker, zip_content)
    mocker.patch(
        "logdetective.server.gitlab.check_artifacts_file_size", return_value=True
    )

    mock_session = mocker.AsyncMock()

    with pytest.raises(
        FileNotFoundError, match="Could not detect failed architecture."
    ):
        await retrieve_and_preprocess_koji_logs(
            gitlab_cfg=gitlab_cfg, job=mock_job, http_session=mock_session
        )


@pytest.mark.asyncio
async def test_raises_logs_too_large_error(mocker: MockerFixture, gitlab_cfg, mock_job):
    """
    Tests that LogsTooLargeError is raised if check_artifacts_file_size returns False.
    """
    mocker.patch(
        "logdetective.server.gitlab.check_artifacts_file_size", return_value=False
    )
    mock_to_thread = mocker.patch("asyncio.to_thread")
    mock_session = mocker.AsyncMock()

    with pytest.raises(LogsTooLargeError):
        await retrieve_and_preprocess_koji_logs(
            gitlab_cfg=gitlab_cfg, job=mock_job, http_session=mock_session
        )

    # Ensure we didn't attempt to download the file
    mock_to_thread.assert_not_called()


@pytest.mark.asyncio
async def test_check_artifacts_file_size(mocker: MockerFixture, gitlab_cfg, mock_job):
    """Test check_artifacts_file_size in case the file doesn't exceed the limit."""
    # Test case where artifact size is within the limit
    mock_session_ok = create_mock_client_response(
        mocker, content_length=gitlab_cfg.max_artifact_size
    )

    result_ok = await check_artifacts_file_size(
        gitlab_cfg=gitlab_cfg, job=mock_job, http_session=mock_session_ok
    )

    assert result_ok is True


@pytest.mark.asyncio
async def test_check_artifacts_file_size_too_large(
    mocker: MockerFixture, gitlab_cfg, mock_job
):
    """Test check_artifacts_file_size in case the file exceeds the limit."""
    mock_session_large = create_mock_client_response(
        mocker, content_length=gitlab_cfg.max_artifact_size + 1
    )

    result_large = await check_artifacts_file_size(
        gitlab_cfg=gitlab_cfg, job=mock_job, http_session=mock_session_large
    )
    assert result_large is False


@pytest.mark.asyncio
async def test_check_artifacts_file_size_handles_http_error(
    mocker: MockerFixture, gitlab_cfg, mock_job
):
    """Test that check_artifacts_file_size raises an HTTPException if the HEAD
    request fails.
    """
    mock_session = create_mock_client_response(mocker)
    # Configure the mock to raise a ClientResponseError, which is what aiohttp does on 4xx/5xx
    mock_session.head.side_effect = aiohttp.ClientResponseError(
        request_info=mocker.Mock(),
        history=mocker.Mock(),
        status=404,
        message="Not Found",
    )

    with pytest.raises(HTTPException, match="Unable to check artifact URL"):
        await check_artifacts_file_size(
            gitlab_cfg=gitlab_cfg, job=mock_job, http_session=mock_session
        )


@pytest.mark.asyncio
async def test_architecture_prioritization(mocker: MockerFixture, gitlab_cfg, mock_job):
    """Test that the correct architecture is chosen when multiple have failed.
    x86_64 should be preferred over aarch64.
    """
    files = {
        "kojilogs/aarch64-build/task_failed.log": "see build.log",
        "kojilogs/aarch64-build/aarch64-build/task_failed.log": "see build.log",
        "kojilogs/aarch64-build/aarch64-build/build.log": "aarch64 failure",
        "kojilogs/aarch64-build/x86_64-build/task_failed.log": "see root.log",
        "kojilogs/aarch64-build/x86_64-build/root.log": "x86_64 failure",
    }
    zip_content = create_zip_archive(files)
    mock_artifact_download(mocker, zip_content)
    mocker.patch(
        "logdetective.server.gitlab.check_artifacts_file_size", return_value=True
    )
    mock_session = mocker.AsyncMock()

    log_url, log_file = await retrieve_and_preprocess_koji_logs(
        gitlab_cfg=gitlab_cfg, job=mock_job, http_session=mock_session
    )

    assert "x86_64-build/root.log" in log_url
    assert log_file.read().decode("utf-8") == "x86_64 failure"


@pytest.mark.asyncio
async def test_toplevel_failure_fallback(mocker: MockerFixture, gitlab_cfg, mock_job):
    """Test that a top-level failure is handled correctly when no specific
    architecture has failed.
    """
    files = {
        "kojilogs/noarch-build/task_failed.log": "Target build already exists",
        "kojilogs/noarch-build/x86_64-build/build.log": "this one didn't fail",
    }
    zip_content = create_zip_archive(files)
    mock_artifact_download(mocker, zip_content)
    mocker.patch(
        "logdetective.server.gitlab.check_artifacts_file_size", return_value=True
    )
    mock_session = mocker.AsyncMock()

    log_url, log_file = await retrieve_and_preprocess_koji_logs(
        gitlab_cfg=gitlab_cfg, job=mock_job, http_session=mock_session
    )

    assert "kojilogs/noarch-build/task_failed.log" in log_url
    assert log_file.read().decode("utf-8") == "Target build already exists"


@pytest.mark.asyncio
async def test_unrecognized_architecture_handling(
    mocker: MockerFixture, gitlab_cfg, mock_job
):
    """Test that if only unrecognized architectures have failed, one is
    chosen alphabetically.
    """
    files = {
        "kojilogs/noarch-build/task_failed.log": "see build.log",
        "kojilogs/noarch-build/a-arch-build/task_failed.log": "see build.log",
        "kojilogs/noarch-build/a-arch-build/build.log": "a-arch failure",
        "kojilogs/noarch-build/b-arch-build/task_failed.log": "see build.log",
        "kojilogs/noarch-build/b-arch-build/build.log": "b-arch failure",
    }
    zip_content = create_zip_archive(files)
    mock_artifact_download(mocker, zip_content)
    mocker.patch(
        "logdetective.server.gitlab.check_artifacts_file_size", return_value=True
    )
    mock_session = mocker.AsyncMock()

    log_url, log_file = await retrieve_and_preprocess_koji_logs(
        gitlab_cfg=gitlab_cfg, job=mock_job, http_session=mock_session
    )

    # Should pick 'a-arch' as it comes first alphabetically
    assert "a-arch-build/build.log" in log_url
    assert log_file.read().decode("utf-8") == "a-arch failure"
