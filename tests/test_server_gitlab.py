import os
import io
from unittest.mock import MagicMock, AsyncMock, patch
import zipfile
import pytest
import pytest_asyncio
import aiohttp
import responses
from aiointercept import aiointercept
from gitlab import Gitlab
from gitlab.exceptions import GitlabError, GitlabGetError
from packaging.version import Version
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from beeai_framework.backend import ChatModel

from logdetective.gitlab import (
    ANSI_ESCAPE_REGEX,
    is_eligible_package,
    retrieve_and_preprocess_koji_logs,
    check_artifacts_file_size,
)
from logdetective.gitlab import process_gitlab_job_event
from logdetective.models import JobHook, GitLabInstanceConfig, APIResponse
from logdetective.database.models import (
    AnalyzeRequestMetrics,
    Forge,
    GitlabMergeRequestJobs,
)
from logdetective.exceptions import LogsTooLargeError, LogDetectiveArtifactsMissingError

from tests.test_helpers import (
    DatabaseFactory,
    create_zip_archive,
    mock_artifact_download,
    mock_job,
    gitlab_cfg,
    create_mock_client_response,
    mock_config,
)


@pytest.mark.parametrize(
    "escape_sequence",
    (
        "\x1b[1m",
        "\x1b[38;5;123m",
        "\x1b[48;2;1;2;3m",
        "\x1b[A",
        "\x1b[12F",
        "\x1b[2K",
        "\x1b[?1049h",
        "\x1b[?1049l",
    ),
)
def test_ansi_escape_regex_removes_csi_sequences(escape_sequence):
    assert ANSI_ESCAPE_REGEX.sub("", f"before{escape_sequence}after") == "beforeafter"


def test_ansi_escape_regex_preserves_text_and_non_escape_controls():
    text = "plain [31m text\b\r"

    assert ANSI_ESCAPE_REGEX.sub("", text) == text


def create_zip_content(filepath) -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(filepath + "task_failed.log", "root.log")
        zip_file.writestr(filepath + "root.log", "ERROR: some sort of error")

    zip_buffer.seek(0)
    return zip_buffer.read()


def set_job_trace(job, content: str) -> None:
    """Attach a streamed trace response to a mock GitLab job."""
    def trace(*, streamed, iterator, chunk_size):
        assert streamed
        assert iterator
        assert chunk_size > 0
        return iter((content.encode("utf-8"),))

    job.trace = trace


async def run_immediately(function, *args, **kwargs):
    """Run blocking work inline in tests that do not exercise thread behavior."""
    return function(*args, **kwargs)


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

    mocked_llm_response = """
{
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello!"
      },
      "finish_reason": "stop"
    }
  ]
}
"""

    with responses.RequestsMock() as sync_rsps:
        async with aiointercept(mock_external_urls=True) as async_rsps:
            async_rsps.head(
                url="https://gitlab.com/api/v4/projects/678/jobs/1/artifacts",
                status=200,
                headers=mocked_headers,
            )
            async_rsps.post(
                url="http://inference-server:8000/v1/chat/completions",
                body=mocked_llm_response,
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


@pytest.fixture
def mock_analysis(request):
    """Fixture to mock analyze_artifacts directly at the server level."""
    message = getattr(request, "param", "This is a mock message")
    mock_response = APIResponse(
        explanation=message,
        snippets=[]
    )
    with patch(
        "logdetective.gitlab.analyze_artifacts",
        AsyncMock(return_value=mock_response)
    ) as mock:
        yield mock


@pytest.mark.parametrize(
    "mock_analysis", ["This is a mock message"], indirect=True
)
@pytest.mark.asyncio
async def test_process_gitlab_job_event(
    mock_config,
    mock_job_hook,
    mock_analysis,
    gitlab_cfg,
):
    async with DatabaseFactory().make_new_db() as session_factory:
        _, _, job_hook = mock_job_hook

        http_session = aiohttp.ClientSession()
        gitlab_connection = Gitlab(
            url="https://gitlab.com/",
        )
        mock_http_session = create_mock_client_response(
            content_length=gitlab_cfg.max_artifact_size
        )

        # We need a chat model for the process_gitlab_job_event
        mock_chat_model = MagicMock(spec=ChatModel)

        await process_gitlab_job_event(
            gitlab_cfg=gitlab_cfg,
            gitlab_connection=gitlab_connection,
            http_session=mock_http_session,
            forge=Forge.gitlab_com,
            job_hook=job_hook,
            chat_model=mock_chat_model,
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
            assert metric.mr_job is not None
            assert metric.mr_job.comment
        await http_session.close()


@pytest.mark.asyncio
async def test_missing_log_data_does_not_post_comment(gitlab_cfg, caplog):
    project = MagicMock(id=42, web_url="https://gitlab.test/pkg")
    project.name = "test-package"
    project.path_with_namespace = "group/test-package"
    job = MagicMock(id=101)
    project.jobs.get.return_value = job
    project.pipelines.get.return_value = MagicMock(
        source="merge_request_event", ref="refs/merge-requests/99/source"
    )
    gitlab_connection = MagicMock()
    gitlab_connection.projects.get.return_value = project
    job_hook = JobHook(
        object_kind="build",
        build_id=101,
        pipeline_id=202,
        build_name="build_centos_stream_rpm",
        build_status="failed",
        project_id=42,
    )
    with (
        patch("logdetective.gitlab.GitlabMergeRequestJobs.get_by_details",
              new_callable=AsyncMock, return_value=None),
        patch("logdetective.gitlab.retrieve_and_preprocess_koji_logs",
              new_callable=AsyncMock,
              side_effect=LogDetectiveArtifactsMissingError("trace unavailable")),
        patch("logdetective.gitlab.analyze_artifacts", new_callable=AsyncMock)
        as analyze,
        patch("logdetective.gitlab.comment_on_mr", new_callable=AsyncMock)
        as comment,
    ):
        response = await process_gitlab_job_event(
            gitlab_cfg, gitlab_connection, AsyncMock(), Forge.gitlab_com,
            job_hook, MagicMock(spec=ChatModel)
        )

    assert response is None
    analyze.assert_not_awaited()
    comment.assert_not_awaited()


@pytest.mark.parametrize(
    ("selected_url", "expected_name"),
    (
        (
            "https://gitlab.test/api/v4/projects/42/jobs/101/artifacts/"
            "kojilogs/x86_64-build/build.log",
            "build.log",
        ),
        (
            "https://gitlab.test/api/v4/projects/42/jobs/101/trace",
            "job-trace.log",
        ),
    ),
)
@pytest.mark.asyncio
async def test_analysis_uses_log_basename(gitlab_cfg, selected_url, expected_name):
    project = MagicMock(id=42, name="test-package", web_url="https://gitlab.test/pkg")
    job = MagicMock(id=101, project_id=42, web_url="https://gitlab.test/jobs/101")
    project.jobs.get.return_value = job
    project.pipelines.get.return_value = MagicMock(
        source="merge_request_event", ref="refs/merge-requests/99/source"
    )
    gitlab_connection = MagicMock()
    gitlab_connection.projects.get.return_value = project
    job_hook = JobHook(
        object_kind="build",
        build_id=101,
        pipeline_id=202,
        build_name="build_centos_stream_rpm",
        build_status="failed",
        project_id=42,
    )
    response = APIResponse(explanation="Build failed", snippets=[])
    with (
        patch("logdetective.gitlab.GitlabMergeRequestJobs.get_by_details",
              new_callable=AsyncMock, return_value=None),
        patch("logdetective.gitlab.retrieve_and_preprocess_koji_logs",
              new_callable=AsyncMock,
              return_value=(selected_url, "raw log")),
        patch("logdetective.gitlab.run_blocking", side_effect=run_immediately),
        patch("logdetective.gitlab.sanitize_artifact", return_value="sanitized log"),
        patch("logdetective.gitlab.add_new_metrics", new_callable=AsyncMock,
              return_value=7),
        patch("logdetective.gitlab.is_eligible_package", return_value=True),
        patch("logdetective.gitlab.analyze_artifacts", new_callable=AsyncMock,
              return_value=response) as analyze,
        patch("logdetective.gitlab.comment_on_mr", new_callable=AsyncMock) as comment,
        patch("logdetective.gitlab.update_metrics", new_callable=AsyncMock),
    ):
        result = await process_gitlab_job_event(
            gitlab_cfg,
            gitlab_connection,
            AsyncMock(),
            Forge.gitlab_com,
            job_hook,
            MagicMock(spec=ChatModel),
        )

    assert result is response
    assert analyze.await_args.kwargs["artifacts"] == {expected_name: "sanitized log"}
    assert comment.await_args.args[4] == selected_url


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
async def test_regression_unknown_arch_logs():
    gl_token = os.environ.get("LD_GITLAB_TOKEN", None)
    data = {
        "name": "Live Network Test Instance",
        "api_url": "https://gitlab.com",
        "api_token": gl_token,
        "max_artifact_size": 120,
    }
    gitlab_cfg = GitLabInstanceConfig.model_validate(data)
    gitlab_connection = Gitlab(
        url=gitlab_cfg.url,
        private_token=gitlab_cfg.api_token,
        timeout=gitlab_cfg.timeout,
    )

    project = gitlab_connection.projects.get(23665037)
    job = project.jobs.get(10226618670)
    mock_session = AsyncMock()
    url, _ = await retrieve_and_preprocess_koji_logs(
        gitlab_cfg=gitlab_cfg, job=job, http_session=mock_session
    )

    assert "task_failed.log" not in url
    assert "build.log" in url
    assert url.startswith(gitlab_cfg.url)


@pytest.mark.asyncio
async def test_fallback_to_task_failed_log_if_no_match(gitlab_cfg, mock_job):
    """
    Tests that if task_failed.log doesn't mention a specific log,
    it returns the task_failed.log itself.
    """
    log_content = "A generic failure occurred. Check Koji for details."
    files = {"kojilogs/x86_64-build/task_failed.log": log_content}
    zip_content = create_zip_archive(files)

    with (
        mock_artifact_download(zip_content),
        patch("logdetective.gitlab.check_artifacts_file_size", return_value=True),
    ):
        mock_session = AsyncMock()
        log_url, log_text = await retrieve_and_preprocess_koji_logs(
            gitlab_cfg=gitlab_cfg, job=mock_job, http_session=mock_session
        )

    assert "artifacts/kojilogs/x86_64-build/task_failed.log" in log_url
    assert log_text == log_content
    assert log_url.startswith(gitlab_cfg.url)


@pytest.mark.asyncio
async def test_trace_is_analyzed_when_no_failure_log(gitlab_cfg, mock_job):
    """The trace is analyzed when the archive has no selected Koji log."""
    files = {
        "kojilogs/x86_64-build/build.log": "This build was successful.",
        "build_logs/upload-to-quay.log": "Failed to convert license (warning)",
    }
    trace_text = "Pipeline setup failed before task logs were created"
    set_job_trace(mock_job, trace_text)
    zip_content = create_zip_archive(files)
    with (
        mock_artifact_download(zip_content),
        patch("logdetective.gitlab.check_artifacts_file_size", return_value=True),
    ):
        mock_session = AsyncMock()
        log_url, log_text = await retrieve_and_preprocess_koji_logs(
            gitlab_cfg=gitlab_cfg,
            job=mock_job,
            http_session=mock_session,
        )

    assert log_url.endswith("/jobs/101/trace")
    assert log_text == trace_text


@pytest.mark.asyncio
async def test_trace_selects_failed_step_log(gitlab_cfg, mock_job):
    failed_path = "build_logs/get-build-target.log"
    failed_url = (
        "https://gitlab.com/api/v4/projects/42/jobs/101/artifacts/" + failed_path
    )
    set_job_trace(
        mock_job,
        "init Succeeded https://gitlab.com/example.log\n"
        f"get-build-target \x1b[91mStepFailed\x1b[0m   {failed_url}\n",
    )
    files = {"build_logs/init.log": "successful", failed_path: "Koji target missing"}
    with (
        mock_artifact_download(create_zip_archive(files)),
        patch("logdetective.gitlab.check_artifacts_file_size", return_value=True),
    ):
        log_url, log_text = await retrieve_and_preprocess_koji_logs(
            gitlab_cfg, mock_job, AsyncMock()
        )

    assert log_url == failed_url
    assert log_text == "Koji target missing"


@pytest.mark.asyncio
async def test_trace_selects_first_failed_architecture(gitlab_cfg, mock_job):
    prefix = "https://gitlab.com/api/v4/projects/42/jobs/101/artifacts/"
    set_job_trace(
        mock_job,
        f"rpmbuild-x86-64 StepFailed {prefix}build_logs/rpmbuild-x86-64.log\n"
        f"rpmbuild-aarch64 StepFailed {prefix}build_logs/rpmbuild-aarch64.log\n",
    )
    files = {
        "build_logs/rpmbuild-x86-64.log": "x86 failure",
        "build_logs/rpmbuild-aarch64.log": "aarch64 failure",
    }
    with (
        mock_artifact_download(create_zip_archive(files)),
        patch("logdetective.gitlab.check_artifacts_file_size", return_value=True),
    ):
        log_url, log_text = await retrieve_and_preprocess_koji_logs(
            gitlab_cfg, mock_job, AsyncMock()
        )

    assert log_url == prefix + "build_logs/rpmbuild-x86-64.log"
    assert log_text == "x86 failure"


@pytest.mark.parametrize("trace_text", (
    "Pipeline setup failed before task logs were created",
    "get-build-target StepFailed https://gitlab.com/api/v4/projects/99/"
    "jobs/101/artifacts/build_logs/get-build-target.log",
    "get-build-target StepFailed https://gitlab.com/api/v4/projects/42/"
    "jobs/101/artifacts/build_logs/missing.log",
))
@pytest.mark.asyncio
async def test_trace_analyzed_when_no_linked_log(gitlab_cfg, mock_job, trace_text):
    set_job_trace(mock_job, trace_text)
    with (
        mock_artifact_download(create_zip_archive({"build_logs/init.log": "done"})),
        patch("logdetective.gitlab.check_artifacts_file_size", return_value=True),
    ):
        log_url, log_text = await retrieve_and_preprocess_koji_logs(
            gitlab_cfg, mock_job, AsyncMock()
        )

    assert log_url == "https://gitlab.com/api/v4/projects/42/jobs/101/trace"
    assert log_text == trace_text


@pytest.mark.asyncio
async def test_oversized_trace_across_chunks_raises(gitlab_cfg, mock_job):
    gitlab_cfg = gitlab_cfg.model_copy(update={"max_artifact_size": 5})

    def oversized_trace(*, streamed, iterator, chunk_size):
        assert streamed
        assert iterator
        assert chunk_size > 0
        return iter((b"123", b"456"))

    mock_job.trace = oversized_trace
    with (
        mock_artifact_download(create_zip_archive({})),
        patch("logdetective.gitlab.check_artifacts_file_size", return_value=True),
        pytest.raises(LogsTooLargeError),
    ):
        await retrieve_and_preprocess_koji_logs(gitlab_cfg, mock_job, AsyncMock())


@pytest.mark.asyncio
async def test_missing_artifacts_falls_back_to_sanitized_trace(gitlab_cfg, mock_job):
    set_job_trace(mock_job, "\x1b[31mFailure\x1b[0m\r\n")
    missing_artifacts = LogDetectiveArtifactsMissingError("artifacts unavailable")
    with (
        patch("logdetective.gitlab.run_blocking", side_effect=run_immediately),
        patch(
            "logdetective.gitlab.check_artifacts_file_size",
            new_callable=AsyncMock,
            side_effect=missing_artifacts,
        ),
    ):
        log_url, log_text = await retrieve_and_preprocess_koji_logs(
            gitlab_cfg, mock_job, AsyncMock()
        )

    assert log_url == "https://gitlab.com/api/v4/projects/42/jobs/101/trace"
    assert log_text == "Failure\n"


@pytest.mark.asyncio
async def test_artifact_download_404_falls_back_to_trace(gitlab_cfg, mock_job):
    set_job_trace(mock_job, "trace after artifact deletion")
    mock_job.artifacts = MagicMock(
        side_effect=GitlabGetError("artifacts were deleted", response_code=404)
    )
    archive = io.BytesIO()
    with (
        patch("logdetective.gitlab.TemporaryFile", return_value=archive),
        patch("logdetective.gitlab.run_blocking", side_effect=run_immediately),
        patch("logdetective.gitlab.check_artifacts_file_size", return_value=True),
    ):
        log_url, log_text = await retrieve_and_preprocess_koji_logs(
            gitlab_cfg, mock_job, AsyncMock()
        )

    assert log_url == "https://gitlab.com/api/v4/projects/42/jobs/101/trace"
    assert log_text == "trace after artifact deletion"
    assert archive.closed


@pytest.mark.asyncio
async def test_artifact_download_non_404_error_propagates(gitlab_cfg, mock_job):
    download_error = GitlabGetError("download failed", response_code=500)
    mock_job.artifacts = MagicMock(side_effect=download_error)
    mock_job.trace = MagicMock()
    archive = io.BytesIO()
    with (
        patch("logdetective.gitlab.TemporaryFile", return_value=archive),
        patch("logdetective.gitlab.run_blocking", side_effect=run_immediately),
        patch("logdetective.gitlab.check_artifacts_file_size", return_value=True),
        pytest.raises(GitlabGetError) as exc_info,
    ):
        await retrieve_and_preprocess_koji_logs(gitlab_cfg, mock_job, AsyncMock())

    assert exc_info.value is download_error
    assert mock_job.trace.call_count == 0
    assert archive.closed


@pytest.mark.parametrize("artifacts_missing", (False, True))
@pytest.mark.asyncio
async def test_unavailable_trace_raises(
    gitlab_cfg, mock_job, artifacts_missing
):
    mock_job.trace = MagicMock(side_effect=GitlabError("trace unavailable"))
    if artifacts_missing:
        with patch(
            "logdetective.gitlab.run_blocking", side_effect=run_immediately
        ):
            with patch(
                "logdetective.gitlab.check_artifacts_file_size",
                new_callable=AsyncMock,
                side_effect=LogDetectiveArtifactsMissingError("artifacts unavailable"),
            ):
                with pytest.raises(LogDetectiveArtifactsMissingError):
                    await retrieve_and_preprocess_koji_logs(
                        gitlab_cfg, mock_job, AsyncMock()
                    )
    else:
        with (
            mock_artifact_download(create_zip_archive({})),
            patch("logdetective.gitlab.check_artifacts_file_size", return_value=True),
        ):
            with pytest.raises(LogDetectiveArtifactsMissingError):
                await retrieve_and_preprocess_koji_logs(
                    gitlab_cfg, mock_job, AsyncMock()
                )


@pytest.mark.asyncio
async def test_empty_artifacts_and_trace_raise(gitlab_cfg, mock_job):
    set_job_trace(mock_job, "")
    with (
        mock_artifact_download(create_zip_archive({})),
        patch("logdetective.gitlab.check_artifacts_file_size", return_value=True),
    ):
        with pytest.raises(LogDetectiveArtifactsMissingError):
            await retrieve_and_preprocess_koji_logs(
                gitlab_cfg, mock_job, AsyncMock()
            )


@pytest.mark.asyncio
async def test_raises_logs_too_large_error(gitlab_cfg, mock_job):
    """
    Tests that LogsTooLargeError is raised if check_artifacts_file_size returns False.
    """
    with (
        patch("logdetective.gitlab.check_artifacts_file_size", return_value=False),
        patch("asyncio.to_thread") as mock_to_thread,
    ):
        mock_session = AsyncMock()
        with pytest.raises(LogsTooLargeError):
            await retrieve_and_preprocess_koji_logs(
                gitlab_cfg=gitlab_cfg, job=mock_job, http_session=mock_session
            )

        # Ensure we didn't attempt to download the file
        mock_to_thread.assert_not_called()


@pytest.mark.parametrize(
    "limit", (512, 1024 + 2), ids=("single-file-size", "cumulative-size")
)
@pytest.mark.asyncio
async def test_raises_logs_too_large_error_decompression(gitlab_cfg, mock_job, limit):
    """
    Tests that LogsTooLargeError is raised if the archive is small, but the decompressed content
    is too big, either one file by itself, or multiple files accumulate over the limit.
    """
    # repetitive payload compresses well, passing the initial Content-Length check
    files = {
        "kojilogs/noarch-build/x86_64-build/task_failed.log": "see build.log",
        "kojilogs/noarch-build/x86_64-build/build.log": "A" * 1024,
    }
    zip_content = create_zip_archive(files)
    with (
        mock_artifact_download(zip_content),
        patch("logdetective.gitlab.check_artifacts_file_size", return_value=True),
    ):
        gitlab_cfg.max_artifact_size = limit
        mock_session = AsyncMock()
        with pytest.raises(LogsTooLargeError, match="exceeded"):
            await retrieve_and_preprocess_koji_logs(
                gitlab_cfg=gitlab_cfg, job=mock_job, http_session=mock_session
            )


@pytest.mark.asyncio
async def test_check_artifacts_file_size(gitlab_cfg, mock_job):
    """Test check_artifacts_file_size in case the file doesn't exceed the limit."""
    # Test case where artifact size is within the limit
    mock_session_ok = create_mock_client_response(
        content_length=gitlab_cfg.max_artifact_size
    )

    result_ok = await check_artifacts_file_size(
        gitlab_cfg=gitlab_cfg, job=mock_job, http_session=mock_session_ok
    )

    assert result_ok is True


@pytest.mark.asyncio
async def test_check_artifacts_file_size_too_large(gitlab_cfg, mock_job):
    """Test check_artifacts_file_size in case the file exceeds the limit."""
    mock_session_large = create_mock_client_response(
        content_length=gitlab_cfg.max_artifact_size + 1
    )

    result_large = await check_artifacts_file_size(
        gitlab_cfg=gitlab_cfg, job=mock_job, http_session=mock_session_large
    )
    assert result_large is False


@pytest.mark.asyncio
async def test_check_artifacts_file_no_content_length(gitlab_cfg, mock_job):
    """Test check_artifacts_file_size evaluates to `False`
    if no content-length header is provided."""
    mock_session_large = create_mock_client_response()

    result_large = await check_artifacts_file_size(
        gitlab_cfg=gitlab_cfg, job=mock_job, http_session=mock_session_large
    )
    assert result_large is False


@pytest.mark.asyncio
async def test_check_artifacts_file_size_handles_http_error(gitlab_cfg, mock_job):
    """Test that check_artifacts_file_size raises an HTTPException if the HEAD
    request fails.
    """
    mock_session = create_mock_client_response(content_length=4096)
    # Configure the mock to raise a ClientResponseError, which is what aiohttp does on 4xx/5xx
    mock_session.head.side_effect = aiohttp.ClientResponseError(
        request_info=MagicMock(),
        history=MagicMock(),
        status=404,
        message="Not Found",
    )

    with pytest.raises(LogDetectiveArtifactsMissingError) as exc_info:
        await check_artifacts_file_size(
            gitlab_cfg=gitlab_cfg, job=mock_job, http_session=mock_session
        )

    assert str(exc_info.value) == (
        f"Requested artifacts are not available - project {mock_job.project_id}, "
        f"job {mock_job.id}, MR {mock_job.project_name}!{mock_job.mr_iid}"
    )


@pytest.mark.asyncio
async def test_architecture_prioritization(gitlab_cfg, mock_job):
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
    with (
        mock_artifact_download(zip_content),
        patch("logdetective.gitlab.check_artifacts_file_size", return_value=True),
    ):
        mock_session = AsyncMock()
        log_url, log_text = await retrieve_and_preprocess_koji_logs(
            gitlab_cfg=gitlab_cfg, job=mock_job, http_session=mock_session
        )

    assert "x86_64-build/root.log" in log_url
    assert log_text == "x86_64 failure"


@pytest.mark.asyncio
async def test_toplevel_failure_fallback(gitlab_cfg, mock_job):
    """Test that a top-level failure is handled correctly when no specific
    architecture has failed.
    """
    files = {
        "kojilogs/noarch-build/task_failed.log": "Target build already exists",
        "kojilogs/noarch-build/x86_64-build/build.log": "this one didn't fail",
    }
    zip_content = create_zip_archive(files)
    with (
        mock_artifact_download(zip_content),
        patch("logdetective.gitlab.check_artifacts_file_size", return_value=True),
    ):
        mock_session = AsyncMock()
        log_url, log_text = await retrieve_and_preprocess_koji_logs(
            gitlab_cfg=gitlab_cfg, job=mock_job, http_session=mock_session
        )

    assert "kojilogs/noarch-build/task_failed.log" in log_url
    assert log_text == "Target build already exists"


@pytest.mark.asyncio
async def test_unrecognized_architecture_handling(gitlab_cfg, mock_job):
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
    with (
        mock_artifact_download(zip_content),
        patch("logdetective.gitlab.check_artifacts_file_size", return_value=True),
    ):
        mock_session = AsyncMock()
        log_url, log_text = await retrieve_and_preprocess_koji_logs(
            gitlab_cfg=gitlab_cfg, job=mock_job, http_session=mock_session
        )

    # Should pick 'a-arch' as it comes first alphabetically
    assert "a-arch-build/build.log" in log_url
    assert log_text == "a-arch failure"
