from pathlib import Path
import asyncio

import aiohttp
import aioresponses
import pytest

from aiolimiter import AsyncLimiter
from fastapi import HTTPException
import gitlab
from flexmock import flexmock

from tests.server.test_helpers import (
    mock_chat_completions,
    MOCK_LOG,
    MOCK_EXPLANATION,
    mock_config,
)

from logdetective.server.config import SERVER_CONFIG
from logdetective.server.llm import perform_staged_analysis, call_llm, perform_analysis
from logdetective.remote_log import RemoteLog
from logdetective.server.config import load_server_config
from logdetective.server.server import initialize_extractors, KojiCallbackManager, ConnectionManager
from logdetective.server.models import (
    Response,
    StagedResponse,
    BuildLogRequest,
    BuildLogFile,
    InferenceConfig,
    Explanation,
    Config,
)
from logdetective.server.utils import get_log_from_payload
from logdetective.utils import sanitize_log


@pytest.mark.asyncio
async def test_loading_config():
    """Load the actual config we have in this repo"""
    # this file - this dir (tests/) - repo root
    repo_root = Path(__file__).parent.parent
    config_file = repo_root / "server" / "config.yml"
    assert load_server_config(str(config_file))


@pytest.mark.asyncio
async def test_process_url():
    url = "http://localhost:8999/"
    mock_header = {"Content-Length": "3"}
    mock_response = "123"
    with aioresponses.aioresponses() as mock:
        mock.head(url, status=200, headers=mock_header)
        mock.get(url, status=200, body=mock_response)
        async with aiohttp.ClientSession() as http:
            remote_log = RemoteLog(url, http)
            url_output_cr = remote_log.process_url()
            url_output = await url_output_cr
            assert url_output == "123"


@pytest.mark.parametrize(
    "mock_chat_completions", ["This is a mock message"], indirect=True
)
@pytest.mark.asyncio
async def test_submit_text_chat_completions(mock_chat_completions):
    # Create InferenceConfig
    inference_cfg = InferenceConfig(
        data={
            "max_tokens": 1000,
            "log_probs": True,
            "url": "http://localhost:8080",
            "api_token": "",
            "model": "stories260K.gguf",
            "temperature": 0.8,
            "max_queue_size": 50,
            "requests_per_minute": 60,
        }
    )
    messages = [
        {
            "role": "user",
            "content": "Hello world!",
        }
    ]
    async_limiter = AsyncLimiter(inference_cfg.requests_per_minute)
    response = await call_llm(
        messages, inference_cfg=inference_cfg, async_request_limiter=async_limiter
    )

    assert isinstance(response, Explanation)
    assert response.text == "This is a mock message"


@pytest.mark.skip(
    reason=(
        "This is a really long unit test,"
        "unskip it when you want to test "
        "the retries mechanism for the submit_text method"
    )
)
@pytest.mark.asyncio
async def test_perform_staged_analysis_with_errors():
    SERVER_CONFIG.inference.url = "http://localhost:8080"
    with aioresponses.aioresponses() as mock:
        mock.post(
            "http://localhost:8080/v1/chat/completions",
            status=504,
            body="Gateway Time-out",
        )
        mock.post(
            "http://localhost:8080/v1/chat/completions",
            status=504,
            body="Gateway Time-out",
        )
        mock.post(
            "http://localhost:8080/v1/chat/completions", status=400, body="Bad Response"
        )
        mock.post(
            "http://localhost:8080/v1/chat/completions", status=400, body="Bad Response"
        )
        async_limiter = AsyncLimiter(SERVER_CONFIG.inference.requests_per_minute)
        extractors = initialize_extractors(SERVER_CONFIG.extractor)
        with pytest.raises(HTTPException):
            await perform_staged_analysis(
                "abc", async_request_limiter=async_limiter, extractors=extractors
            )


@pytest.mark.parametrize("mock_chat_completions", [MOCK_EXPLANATION], indirect=True)
@pytest.mark.asyncio
async def test_perform_analysis(
    mock_chat_completions,
):
    async_limiter = AsyncLimiter(100)
    extractors = initialize_extractors(SERVER_CONFIG.extractor)
    result = await perform_analysis(
        MOCK_LOG, async_request_limiter=async_limiter, extractors=extractors
    )

    assert result.explanation.text == MOCK_EXPLANATION


@pytest.mark.parametrize("mock_chat_completions", [MOCK_EXPLANATION], indirect=True)
@pytest.mark.asyncio
async def test_perform_staged_analysis(
    mock_chat_completions,
):
    async_limiter = AsyncLimiter(100)
    extractors = initialize_extractors(SERVER_CONFIG.extractor)

    result = await perform_staged_analysis(
        MOCK_LOG, async_request_limiter=async_limiter, extractors=extractors
    )

    assert result.explanation.text == MOCK_EXPLANATION


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

    assert isinstance(connection_manager.gitlab_connections["https://gitlab.com"], gitlab.Gitlab)
    assert not connection_manager.gitlab_http_sessions["https://gitlab.com"].closed

    await connection_manager.close()
    assert connection_manager.gitlab_http_sessions["https://gitlab.com"].closed


@pytest.mark.parametrize(
    "mock_chat_completions, analysis_func, expected_type",
    [
        (MOCK_EXPLANATION, perform_analysis, Response),
        (MOCK_EXPLANATION, perform_staged_analysis, StagedResponse),
    ],
    indirect=["mock_chat_completions"]
)
@pytest.mark.asyncio
async def test_analyze_with_files_payload(mock_chat_completions, analysis_func, expected_type):
    payload = BuildLogRequest(
        files=[
            BuildLogFile(name="build.log", content=MOCK_LOG)
        ]
    )

    async with aiohttp.ClientSession() as session:
        log_text = await get_log_from_payload(payload, session)

    log_text = sanitize_log(log_text)

    async_limiter = AsyncLimiter(100)
    extractors = initialize_extractors(SERVER_CONFIG.extractor)
    result = await analysis_func(
        log_text,
        async_request_limiter=async_limiter,
        extractors=extractors
    )

    assert isinstance(result, expected_type)
    assert result.explanation.text == MOCK_EXPLANATION


@pytest.mark.asyncio
async def test_get_log_from_payload_with_files():
    payload = BuildLogRequest(
        files=[
            BuildLogFile(name="test.log", content=MOCK_LOG),
            BuildLogFile(name="ignored.log", content="This should be ignored")
        ]
    )

    async with aiohttp.ClientSession() as session:
        log_text = await get_log_from_payload(payload, session)

    assert log_text == MOCK_LOG


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dirty_log, redacted_value",
    [
        ("RSA key 00112233AABBCCDD", "00112233AABBCCDD"),
        ("pubkey-deadbeef-cafe0123", "deadbeef-cafe0123"),
        ("Commited 2020 example@mail.com", "example@mail.com"),
    ],
    indirect=False
)
async def test_get_log_from_payload_files_sanitization(dirty_log, redacted_value):
    payload = BuildLogRequest(
        files=[BuildLogFile(name="test.log", content=dirty_log)]
    )

    async with aiohttp.ClientSession() as session:
        log_text = await get_log_from_payload(payload, session)

    assert log_text == dirty_log

    sanitized = sanitize_log(log_text)

    assert sanitized != dirty_log
    assert redacted_value not in sanitized
    assert any(i in sanitized for i in ["FFFF", "ffff", "copr-team"])


@pytest.mark.asyncio
async def test_get_log_from_payload_url_sanitization():
    dirty_log = "This email should be sanitized: contact@someone.com"
    payload = BuildLogRequest(url="http://path.to/file.log")
    awaited_dirty_log = asyncio.Future()
    awaited_dirty_log.set_result(dirty_log)
    mock_remote_log = flexmock(RemoteLog)

    async with aiohttp.ClientSession() as session:
        mock_remote_log.should_receive("__init__").with_args("http://path.to/file.log", session)
        mock_remote_log.should_receive("process_url").and_return(awaited_dirty_log)
        log_text = await get_log_from_payload(payload, session)

    assert log_text == dirty_log

    sanitized = sanitize_log(log_text)

    assert sanitized != dirty_log
    assert "contact@someone.com" not in sanitized
    assert "copr-team@redhat.com" in sanitized
