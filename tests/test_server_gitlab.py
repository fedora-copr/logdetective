import io
import zipfile
import pytest
import pytest_asyncio
import aiohttp
import responses
import aioresponses
from packaging.version import Version

from aiolimiter import AsyncLimiter
from flexmock import flexmock

from test_helpers import (
    DatabaseFactory,
)

from logdetective.server.gitlab import is_eligible_package
from logdetective.server.server import process_gitlab_job_event
from logdetective.server.models import JobHook, GitLabInstanceConfig, Config
from logdetective.server.compressors import RemoteLogCompressor
from logdetective.server.database.models import (
    AnalyzeRequestMetrics,
    Forge,
)
from logdetective.server import gitlab, llm


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
                "api_endpoint": "/chat/completitions",
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


@pytest.mark.skipif(
    Version(aioresponses.__version__) < Version("0.7.8"),
    reason="aioresponses lacks support for base_url",
)
@pytest.mark.asyncio
async def test_process_gitlab_job_event(mock_config, mock_job_hook):
    with DatabaseFactory().make_new_db() as session_factory:
        _, _, job_hook = mock_job_hook
        gitlab_instance = GitLabInstanceConfig(
            name="mocked_gitlab",
            data={
                "url": "https://gitlab.com",
                "api_path": "/api/v4",
                "api_token": "empty",
                "max_artifact_size": 300,
            },
        )
        http_session = aiohttp.ClientSession()
        await process_gitlab_job_event(
            gitlab_cfg=gitlab_instance,
            forge=Forge.gitlab_com,
            job_hook=job_hook,
        )
        with session_factory() as db_session:
            metric = db_session.get(AnalyzeRequestMetrics, 1)
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
