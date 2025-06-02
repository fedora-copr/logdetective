import json
from pathlib import Path
from packaging.version import Version

import aiohttp
import aioresponses
import pytest

from fastapi import HTTPException

from logdetective.server.config import SERVER_CONFIG
from logdetective.server.llm import (
    perform_staged_analysis,
    submit_text,
)
from logdetective.remote_log import RemoteLog
from logdetective.server.config import load_server_config
from logdetective.server.models import InferenceConfig, Explanation

from tests.server.test_helpers import mock_chat_completions


@pytest.mark.asyncio
async def test_loading_config():
    """Load the actual config we have in this repo"""
    # this file - this dir (tests/) - repo root
    repo_root = Path(__file__).parent.parent
    config_file = repo_root / "server" / "config.yml"
    assert load_server_config(str(config_file))


@pytest.mark.asyncio
async def test_process_url():
    mock_response = "123"
    with aioresponses.aioresponses() as mock:
        mock.get("http://localhost:8999/", status=200, body=mock_response)
        async with aiohttp.ClientSession() as http:
            remote_log = RemoteLog("http://localhost:8999/", http)
            url_output_cr = remote_log.process_url()
            url_output = await url_output_cr
            assert url_output == "123"


@pytest.mark.parametrize("mock_chat_completions", ["This is a mock message"], indirect=True)
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

    response = await submit_text("Hello world!", inference_cfg=inference_cfg)

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
        with pytest.raises(HTTPException):
            await perform_staged_analysis("abc")
