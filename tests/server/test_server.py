from pathlib import Path
import asyncio

import aiohttp
import aioresponses
import pytest
from pydantic import HttpUrl

import gitlab
from flexmock import flexmock

from logdetective.server.config import SERVER_CONFIG
from logdetective.remote_log import RemoteLog
from logdetective.server.config import load_server_config
from logdetective.server.server import KojiCallbackManager, ConnectionManager
from logdetective.server.models import (
    AnalysisRequest,
    ArtifactFile,
    RemoteArtifactFile,
    Config,
)
from logdetective.server.utils import get_artifacts_from_payload
from logdetective.utils import sanitize_artifact

from tests.server.test_helpers import (
    MOCK_LOG,
    mock_config,
)


@pytest.mark.asyncio
async def test_loading_config():
    """Load the actual config we have in this repo"""
    # this file - this dir (tests/) - repo root
    repo_root = Path(__file__).parent.parent
    config_file = repo_root / "server" / "config.yml"
    assert load_server_config(str(config_file))


@pytest.mark.asyncio
async def test_get_url_content():
    url = "http://localhost:8999/"
    mock_header = {"Content-Length": "3"}
    mock_response = "123"
    with aioresponses.aioresponses() as mock:
        mock.head(url, status=200, headers=mock_header)
        mock.get(url, status=200, body=mock_response)
        async with aiohttp.ClientSession() as http:
            remote_log = RemoteLog(url, http)
            url_output_cr = remote_log.get_url_content()
            url_output = await url_output_cr
            assert url_output == "123"


def test_koji_callback_manager():
    """Test KojiCallbackManager initialization, callback registration
    and removal."""
    manager = KojiCallbackManager()
    assert len(manager.get_callbacks(1)) == 0

    manager.register_callback(1, "test_callback")

    assert len(manager.get_callbacks(1)) == 1

    manager.clear_callbacks(1)

    assert len(manager.get_callbacks(1)) == 0


@pytest.mark.asyncio
async def test_connection_manager(mock_config):
    """Test that ConnectionManager can handle initialization and disposal
    of managed objects, sessions in particular."""
    connection_manager = ConnectionManager()
    server_config = mock_config["server_config"]
    assert isinstance(server_config, Config)

    assert len(server_config.gitlab.instances) == 1

    await connection_manager.initialize(server_config)
    assert len(connection_manager.gitlab_connections) == 1
    assert len(connection_manager.gitlab_http_sessions) == 1

    assert isinstance(
        connection_manager.gitlab_connections["https://gitlab.com"], gitlab.Gitlab
    )
    assert not connection_manager.gitlab_http_sessions["https://gitlab.com"].closed

    await connection_manager.close()
    assert connection_manager.gitlab_http_sessions["https://gitlab.com"].closed


@pytest.mark.parametrize("request_size", [0, 10, 2000])
@pytest.mark.asyncio
async def test_get_log_from_payload_with_files(request_size):
    payload = AnalysisRequest(
        files=[
            ArtifactFile(name="test.log", content=MOCK_LOG),
            ArtifactFile(name="ignored.log", content=MOCK_LOG),
        ]
    )

    async with aiohttp.ClientSession() as session:
        artifacts = await get_artifacts_from_payload(
            payload, session, request_size=request_size
        )

    assert "test.log" in artifacts
    assert "ignored.log" in artifacts
    assert len(artifacts) == 2

    for log_text in artifacts.values():
        assert log_text == MOCK_LOG


@pytest.mark.asyncio
@pytest.mark.parametrize("request_size", [0, 10, 2000])
@pytest.mark.parametrize(
    "dirty_log, redacted_value",
    [
        ("RSA key 00112233AABBCCDD", "00112233AABBCCDD"),
        ("pubkey-deadbeef-cafe0123", "deadbeef-cafe0123"),
        ("Commited 2020 example@mail.com", "example@mail.com"),
    ],
    indirect=False,
)
async def test_get_log_from_payload_files_sanitization(
    dirty_log, redacted_value, request_size
):
    payload = AnalysisRequest(files=[ArtifactFile(name="test.log", content=dirty_log)])

    assert payload.files is not None
    async with aiohttp.ClientSession() as session:
        artifacts = await get_artifacts_from_payload(
            payload, session, request_size=request_size
        )

    assert len(artifacts) == 1
    assert "test.log" in artifacts
    assert artifacts[payload.files[0].name] == dirty_log

    for text in artifacts.values():
        sanitized = sanitize_artifact(text)

        assert sanitized != dirty_log
        assert redacted_value not in sanitized
        assert any(i in sanitized for i in ["FFFF", "ffff", "copr-team"])


@pytest.mark.parametrize("request_size", [0, 10, 2000])
@pytest.mark.asyncio
async def test_get_log_from_payload_url_sanitization(request_size):
    dirty_log = "This email should be sanitized: contact@someone.com"
    payload = AnalysisRequest(
        files=[
            RemoteArtifactFile(
                name="mock_log.log", url=HttpUrl("http://path.to/file.log")
            )
        ]
    )
    awaited_dirty_log = asyncio.Future()
    awaited_dirty_log.set_result(dirty_log)
    mock_remote_log = flexmock(RemoteLog)

    async with aiohttp.ClientSession() as session:
        mock_remote_log.should_receive("__init__").with_args(
            "http://path.to/file.log",
            session,
            limit_bytes=SERVER_CONFIG.general.max_artifact_size - request_size,
        )
        mock_remote_log.should_receive("get_url_content").and_return(awaited_dirty_log)
        artifacts = await get_artifacts_from_payload(
            payload, session, request_size=request_size
        )

    assert len(artifacts) == 1
    assert "mock_log.log" in artifacts

    assert artifacts["mock_log.log"] != dirty_log
    assert "contact@someone.com" not in artifacts["mock_log.log"]
    assert "copr-team@redhat.com" in artifacts["mock_log.log"]
