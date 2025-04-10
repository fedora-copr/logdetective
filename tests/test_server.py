import os
import asyncio
import subprocess
from contextlib import contextmanager
from pathlib import Path
from time import sleep

import pytest
import sys
from huggingface_hub import hf_hub_download

from logdetective.server.server import submit_to_llm_endpoint, process_url
from logdetective.server.utils import load_server_config


@contextmanager
def start_server():
    """ Context manager for running a llama-server.
    The binary is set via env var LLAMA_CPP_SERVER_BINARY"""
    model_hf_repo = "ggml-org/models"
    model_hf_file = "tinyllamas/stories260K.gguf"
    path = hf_hub_download(repo_id=model_hf_repo, filename=model_hf_file)
    binary_path = os.environ["LLAMA_CPP_SERVER_BINARY"]
    if not os.path.isfile(binary_path):
        raise FileNotFoundError(f"Llama-server binary does not exist: {binary_path}")
    process = subprocess.Popen(
        [binary_path, "--host", "localhost", "--port",
         "8080", "--model", path])
    yield
    process.kill()


def test_loading_config():
    """ Load the actual config we have in this repo """
    # this file - this dir (tests/) - repo root
    repo_root = Path(__file__).parent.parent
    config_file = repo_root / "server" / "config.yml"
    assert load_server_config(str(config_file))


@pytest.mark.skip
def test_submit_to_llm():
    """ Test async communication with an OpenAI compat inference server """
    with start_server():
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
        response = submit_to_llm_endpoint(url, data, headers, False)
        response = asyncio.run(response)
    assert response


def test_process_url():
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "http.server", "-d", "/tmp", "8999"])
        # let's give the server time to boot up
        sleep(0.05)
        dir_listing_cr = process_url("http://localhost:8999/")
        dir_listing = asyncio.run(dir_listing_cr)
        assert dir_listing
    finally:
        process.kill()
