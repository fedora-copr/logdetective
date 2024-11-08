Log Detective
=============

[![PyPI - Version](https://img.shields.io/pypi/v/logdetective?color=blue)][PyPI Releases]

[PyPI Releases]: https://pypi.org/project/logdetective/#history

A Python tool to analyze logs using a Language Model (LLM) and Drain template miner.

Installation
------------

**Fedora 40+**

    dnf install logdetective

**From Pypi repository**

The logdetective project is published on the the the the the [Pypi repository](https://pypi.org/project/logdetective/). The `pip` tool can be used for installation.

First, ensure that the necessary dependencies for the `llama-cpp-python` project are installed. For Fedora, install `gcc-c++`:

    # for Fedora it will be:
    dnf install gcc-c++

Then, install the `logdetective` project using pip:

    # then install logdetective project
    pip install logdetective

**Local repository install**

    pip install .

Usage
-----

To analyze a log file, run the script with the following command line arguments:
- `url` (required): The URL of the log file to be analyzed.
- `--model` (optional, default: "Mistral-7B-Instruct-v0.2-GGUF"): The path or URL of the language model for analysis. As we are using LLama.cpp we want this to be in the `gguf` format. You can include the download link to the model here. If the model is already on your machine it will skip the download.
- `--summarizer` (optional, default: "drain"): Choose between LLM and Drain template miner as the log summarizer. You can also provide the path to an existing language model file instead of using a URL.
- `--n_lines` (optional, default: 8): The number of lines per chunk for LLM analysis. This only makes sense when you are summarizing with LLM.
- `--n_clusters` (optional, default 8): Number of clusters for Drain to organize log chunks into. This only makes sense when you are summarizing with Drain

Example usage:

    logdetective https://example.com/logs.txt

Or if the log file is stored locally:

    logdetective ./data/logs.txt

Example you want to use a different model:

    logdetective https://example.com/logs.txt --model https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q5_K_S.gguf?download=true
    logdetective https://example.com/logs.txt --model QuantFactory/Meta-Llama-3-8B-Instruct-GGUF

Note that streaming with some models (notably Meta-Llama-3 is broken) is broken and can be workarounded by `no-stream` option:

    logdetective https://example.com/logs.txt --model QuantFactory/Meta-Llama-3-8B-Instruct-GGUF --no-stream


Real Example
------------
Let's have a look at a real world example. Log Detective can work with any logs though we optimize it for build logs.

We're going to analyze a failed build of a python-based library that happened in Fedora Koji buildsystem:
```
$ logdetective https://kojipkgs.fedoraproject.org//work/tasks/8157/117788157/build.log
Explanation:
[Child return code was: 0] : The rpm build process executed successfully without any errors until the 'check' phase.

[wamp/test/test_wamp_component_aio.py::test_asyncio_component] : Pytest found
two tests marked with '@pytest.mark.asyncio' but they are not async functions.
This warning can be ignored unless the tests are intended to be run
asynchronously.

[wamp/test/test_wamp_component_aio.py::test_asyncio_component_404] : Another
Pytest warning for the same issue as test_asyncio_component.

[-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html] :
This line is not related to the error, but it is a reminder to refer to Pytest
documentation for handling warnings.

[=========================== short test summary info
============================] : This section shows the summary of tests that
were executed. It shows the number of tests passed, failed, skipped,
deselected, and warnings.

[FAILED wamp/test/test_wamp_cryptosign.py::TestSigVectors::test_vectors] : A
failed test is reported with the name of the test file, the name of the test
method, and the name of the test case that failed. In this case,
TestSigVectors::test_vectors failed.

[FAILED
websocket/test/test_websocket_protocol.py::WebSocketClientProtocolTests::test_auto_ping]
: Another failed test is reported with the same format as the previous test. In
this case, it is WebSocketClientProtocolTests::test_auto_ping that failed.

[FAILED websocket/test/test_websocket_protocol.py::WebSocketServerProtocolTests::test_interpolate_server_status_template]
: A third failed test is reported with the same format as the previous tests.
In this case, it is
WebSocketServerProtocolTests::test_interpolate_server_status_template that
failed.

[FAILED websocket/test/test_websocket_protocol.py::WebSocketServerProtocolTests::test_sendClose_reason_with_no_code]
: Another failed test is reported. This time it is
WebSocketServerProtocolTests::test_sendClose_reason_with_no_code.

[FAILED websocket/test/test_websocket_protocol.py::WebSocketServerProtocolTests::test_sendClose_str_reason]
: Another failed test is reported with the same test file and test method name,
but a different test case name: test_sendClose_str_reason.

[==== 13 failed, 195 passed, 64 skipped, 13 deselected, 2 warnings in 6.55s
=====] : This is the summary of all tests that were executed, including the
number of tests that passed, failed, were skipped, deselected, or produced
warnings. In this case, there were 13 failed tests among a total of 211 tests.

[error: Bad exit status from /var/tmp/rpm-tmp.8C0L25 (%check)] : An error
message is reported indicating that the 'check' phase of the rpm build process
failed with a bad exit status.
```

It looks like a wall of text. Similar to any log. The main difference is that here we have the most significant lines of a logfile wrapped in `[ ] : ` and followed by textual explanation of the log text done by mistral 7b.


Contributing
------------

Contributions are welcome! Please submit a pull request if you have any improvements or new features to add. Make sure your changes pass all existing tests before submitting.

To develop logdetective, you should fork this repository, clone your fork, and install dependencies using pip:

    git clone https://github.com/yourusername/logdetective.git
    cd logdetective
    pip install .

Make changes to the code as needed and run pre-commit.

Tests
-----

The [tox](https://github.com/tox-dev/tox) is used to manage tests. Please install `tox` package into your distribution and run:

    tox

This will create a virtual environment with dependencies and run all the tests. For more information follow the tox help.

To run only a specific test execute this:

    tox run -e style # to run flake8

or

    tox run -e lint # to run pylint

Server
------

FastApi based server is implemented in `logdetective/server.py`. In order to run it in a development mode,
simply start llama-cpp-python server with your chosen model as described in llama-cpp-python [docs](https://llama-cpp-python.readthedocs.io/en/latest/server/#running-the-server).

Afterwards, start the logdetective server with `fastapi dev logdetective/server.py --port 8080`.
Requests can then be made with post requests, for example:

    curl --header "Content-Type: application/json" --request POST --data '{"url":"<YOUR_URL_HERE>"}' http://localhost:8080/analyze

For more accurate responses, you can use `/analyze/staged` endpoint. This will submit snippets to model for individual analysis first.
Afterwards the model outputs are used to construct final prompt. This will take substantially longer, compared to plain `/analyze`

    curl --header "Content-Type: application/json" --request POST --data '{"url":"<YOUR_URL_HERE>"}' http://localhost:8080/analyze/staged

We also have a Containerfile and composefile to run the logdetective server and llama server in containers.

Before doing `podman-compose up`, make sure to set `MODELS_PATH` environment variable and point to a directory with your local model files:
```
$ export MODELS_PATH=/path/to/models/
$ ll $MODELS_PATH
-rw-r--r--. 1 tt tt 3.9G apr 10 17:18  mistral-7b-instruct-v0.2.Q4_K_S.gguf
```

If the variable is not set, `./models` is mounted inside by default.

Model can be downloaded from [our Hugging Space](https://huggingface.co/fedora-copr) by:
```
$ curl -L -o models/mistral-7b-instruct-v0.2.Q4_K_S.gguf https://huggingface.co/fedora-copr/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/ggml-model-Q4_K_S.gguf
```


License
-------

This project is licensed under the Apache-2.0 License - see the LICENSE file for details.
