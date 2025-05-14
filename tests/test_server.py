import json
from pathlib import Path

import aiohttp
import aioresponses
import pytest

from fastapi import HTTPException
from aiolimiter import AsyncLimiter

from logdetective.server.config import SERVER_CONFIG
from logdetective.server.llm import (
    perform_staged_analysis,
    submit_to_llm_endpoint,
    submit_text_chat_completions,
)
from logdetective.remote_log import RemoteLog
from logdetective.server.config import load_server_config


def test_loading_config():
    """Load the actual config we have in this repo"""
    # this file - this dir (tests/) - repo root
    repo_root = Path(__file__).parent.parent
    config_file = repo_root / "server" / "config.yml"
    assert load_server_config(str(config_file))


@pytest.mark.asyncio
async def test_submit_to_llm():
    """Test async communication with an OpenAI compat inference server"""
    mock_response = {
        "choices": [
            {
                "finish_reason": "length",
                "index": 0,
                "message": {"role": "assistant", "content": "So she was very..."},
                "logprobs": {
                    "content": [
                        {
                            "id": 437,
                            "token": "S",
                            "bytes": [83],
                            "logprob": -3.2835686206817627,
                            "top_logprobs": [
                                {
                                    "id": 436,
                                    "token": '"',
                                    "bytes": [34],
                                    "logprob": -1.1505162715911865,
                                }
                            ],
                        },
                        {
                            "id": 414,
                            "token": "o",
                            "bytes": [111],
                            "logprob": -2.8719468116760254,
                            "top_logprobs": [
                                {
                                    "id": 425,
                                    "token": "u",
                                    "bytes": [117],
                                    "logprob": -0.44975802302360535,
                                }
                            ],
                        },
                    ]
                },
                "created": 1744381795,
                "model": "stories260K.gguf",
                "system_fingerprint": "b4839-42994048",
                "object": "chat.completion",
                "usage": {
                    "completion_tokens": 1000,
                    "prompt_tokens": 45,
                    "total_tokens": 1045,
                },
                "id": "chatcmpl - PreWjyi55gzlFy91x0LEfj7Xp91G197i",
                "timings": {
                    "prompt_n": 1,
                    "prompt_ms": 6.191,
                    "prompt_per_token_ms": 6.191,
                    "prompt_per_second": 161.52479405588758,
                    "predicted_n": 1000,
                    "predicted_ms": 1137.463,
                    "predicted_per_token_ms": 1.137463,
                    "predicted_per_second": 879.1494756312953,
                },
            }
        ]
    }
    with aioresponses.aioresponses() as mock:
        mock.post(
            "http://localhost:8080/v1/chat/completions",
            status=200,
            body=json.dumps(mock_response),
        )
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello!",
                }
            ],
            "max_tokens": 1000,
            "logprobs": 1,
            "stream": False,
            "model": "stories260K.gguf",
            "temperature": 0.8,
        }
        headers = {"Content-Type": "application/json"}
        url = "http://localhost:8080/v1/chat/completions"

        async with aiohttp.ClientSession() as http:
            response = submit_to_llm_endpoint(http, url, data, headers, False)
            response = await response
        assert response == mock_response


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


@pytest.mark.asyncio
async def test_submit_text_chat_completions():
    mock_response = b"123"
    SERVER_CONFIG.inference.url = "http://localhost:8080"
    with aioresponses.aioresponses() as mock:
        mock.post(
            "http://localhost:8080/v1/chat/completions", status=200, body=mock_response
        )
        async with aiohttp.ClientSession() as http:
            response = await submit_text_chat_completions(http, "asd", {}, stream=True)
            async for x in response.content:
                assert x == mock_response


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
        async with aiohttp.ClientSession() as http:
            with pytest.raises(HTTPException):
                await perform_staged_analysis(http, "abc")


@pytest.mark.asyncio
async def test_submit_text_strange_log():
    SERVER_CONFIG.inference.url = (
        "https://mistral-7b-instruct-v0-2-lqmssh22.apps.int.stc.ai.prod.us-east-1.aws.paas.redhat.com"
    )
    async with aiohttp.ClientSession() as http:
        with open("tests/data/strange-snippet-x86_64-root.test-log", "r") as log:
            content = log.read()
            staged_response = await perform_staged_analysis(http, log_text=content)
            assert staged_response
