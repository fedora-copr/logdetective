Log Detective
=============

[![PyPI - Version](https://img.shields.io/pypi/v/logdetective?color=blue)][PyPI Releases]

[PyPI Releases]: https://pypi.org/project/logdetective/#history

A tool, service and RHEL process integration to analyze logs using a Large Language Model (LLM) and a [Drain template miner](https://github.com/logpai/Drain3).

The service that explains logs is available here: https://logdetective.com/explain

Note: if you are looking for code of website logdetective.com it is in [github.com/fedora-copr/logdetective-website](https://github.com/fedora-copr/logdetective-website).

Installation
------------

**Fedora 41+**

    dnf install logdetective

**From Pypi repository**

The logdetective project is published on the [Pypi repository](https://pypi.org/project/logdetective/). The `pip` tool can be used for installation.

First, ensure that the necessary dependencies for the `llama-cpp-python` project are installed. For Fedora, install `gcc-c++`:

    # for Fedora it will be:
    dnf install gcc-c++

Then, install the `logdetective` project using pip:

    pip install logdetective

**Local repository install**

Clone this repository and install with pip:

    pip install .

Usage
-----

To analyze a log file, run the script with the following command line arguments:
- `url` (required): The URL of the log file to be analyzed.
- `--model` (optional, default: "Mistral-7B-Instruct-v0.2-GGUF"): The path or URL of the language model for analysis. As we are using LLama.cpp we want this to be in the `gguf` format. You can include the download link to the model here. If the model is already on your machine it will skip the download.
- `--summarizer` DISABLED: LLM summarization option was removed. Argument is kept for backward compatibility only.(optional, default: "drain"): Choose between LLM and Drain template miner as the log summarizer. You can also provide the path to an existing language model file instead of using a URL.
- `--n_lines` DISABLED: LLM summarization option was removed. Argument is kept for backward compatibility only. (optional, default: 8): The number of lines per chunk for LLM analysis. This only makes sense when you are summarizing with LLM.
- `--n_clusters` (optional, default 8): Number of clusters for Drain to organize log chunks into. This only makes sense when you are summarizing with Drain

Example usage:

    logdetective https://example.com/logs.txt

Or if the log file is stored locally:

    logdetective ./data/logs.txt

Example you want to use a different model:

    logdetective https://example.com/logs.txt --model https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q5_K_S.gguf?download=true
    logdetective https://example.com/logs.txt --model QuantFactory/Meta-Llama-3-8B-Instruct-GGUF

Example of different suffix (useful for models that were quantized)

    logdetective https://kojipkgs.fedoraproject.org//work/tasks/3367/131313367/build.log --model 'fedora-copr/granite-3.2-8b-instruct-GGUF' -F Q4_K.gguf

Example of altered prompts:

     cp ~/.local/lib/python3.13/site-packages/logdetective/prompts.yml ~/my-prompts.yml
     vi ~/my-prompts.yml # edit the prompts there to better fit your needs
     logdetective https://kojipkgs.fedoraproject.org//work/tasks/3367/131313367/build.log --prompts ~/my-prompts.yml


Note that streaming with some models (notably Meta-Llama-3 is broken) is broken and can be worked around by `no-stream` option:

    logdetective https://example.com/logs.txt --model QuantFactory/Meta-Llama-3-8B-Instruct-GGUF --no-stream


Real Example
------------
Let's have a look at a real world example. Log Detective can work with any logs though we optimize it for RPM build logs.

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
For bigger code changes, please consult us first by creating an issue.

We are always looking for more annotated snippets that will increase the quality of Log Detective's results. The contributions happen in our website: https://logdetective.com/

Log Detective performs several inference queries while evaluating a log file. Prompts are stored in a separate file (more info below: https://github.com/fedora-copr/logdetective?tab=readme-ov-file#system-prompts). If you have an idea for improvements to our prompts, please open a PR and we'd happy to test it out.

To develop Log Detective, you should fork this repository, clone your fork, and install dependencies using pip:

    git clone https://github.com/yourusername/logdetective.git
    cd logdetective
    pip install .

Make changes to the code as needed and run pre-commit.

Tests
-----

Tests for code used by server must placed in the `./tests/server/` path, while tests for general
code must be in the `./tests/base/` path.

The [tox](https://github.com/tox-dev/tox) is used to manage tests. Please install `tox` package into your distribution and run:

    tox

This will create a virtual environment with dependencies and run all the tests. For more information follow the tox help.

To run only a specific test execute this:

    tox run -e style # to run flake8

or

    tox run -e lint # to run pylint

Tox environments for base and server tests are separate, each installs different dependencies.

Running base tests:

    tox run -e pytest_base

Running server tests:

    tox run -e pytest_server

To run server test suite you will need postgresql client utilities.

    dnf install postgresql

Visual Studio Code testing with podman/docker-compose
-----------------------------------------------------

- In `Containerfile`, add `debugpy` as a dependency

```diff
-RUN pip3 install llama_cpp_python==0.2.85 sse-starlette starlette-context \
+RUN pip3 install llama_cpp_python==0.2.85 sse-starlette starlette-context debugpy\
```

- Rebuild server image with new dependencies

```
make rebuild-server
```

- Forward debugging port in `docker-compose.yaml` for `server` service.

```diff
     ports:
       - "${LOGDETECTIVE_SERVER_PORT:-8080}:${LOGDETECTIVE_SERVER_PORT:-8080}"
+      - "${VSCODE_DEBUG_PORT:-5678}:${VSCODE_DEBUG_PORT:-5678}"
```

- Add `debugpy` code in a logdetective file where you want to stop at first.

```diff
+import debugpy
+debugpy.listen(("0.0.0.0", 5678))
+debugpy.wait_for_client()
```

- Prepare `.vscode/lunch.json` configuration for Visual Studio Code (at least the following configuration is needed)

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python Debugger: Remote Attach",
      "type": "debugpy",
      "request": "attach",
      "connect": {
        "host": "localhost",
        "port": 5678
      },
      "pathMappings": [
        {
          "localRoot": "${workspaceFolder}",
          "remoteRoot": "/src"
        }
      ]
    }
  ]
}
```

- Run the server

```
podman-compose up server
```

- Run Visual Stdio Code debug configuration named *Python Debug: Remote Attach*

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

Generate a new database revision with alembic
---------------------------------------------

Modify the database models (`logdetective/server/database/model.py).

Generate a new database revision with the command:

**Warning**: this command will start up a new server
and shut it down when the operation completes.

```
CHANGE="A change comment" make alembic-generate-revision
```

Our production instance
-----------------------

Our FastAPI server and model inference server run through `podman-compose` on an
Amazon AWS intance. The VM is provisioned by an
[ansible playbook](https://pagure.io/fedora-infra/ansible/blob/main/f/roles/logdetective/tasks/main.yml).

You can control the server through:

```
cd /root/logdetective
podman-compose -f docker-compose-prod.yaml ...
```

The `/root` directory contains valuable data. If moving to a new instance,
please backup the whole directory and transfer it to the new instance.

Fore some reason, we need to manually run this command after every reboot:

```
nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

HTTPS certificate generated through:

```
certbot certonly --standalone -d logdetective01.fedorainfracloud.org
```

Certificates need to be be placed into location specified by the`LOGDETECTIVE_CERTDIR`
env var and the service should be restarted.

Querying statistics
-------------------

You can retrieve statistics about server requests and responses over a specified time period
using either a browser, the `curl` or the `http` command (provided by the `httpie` package).

When no time period is specified, the query defaults to the last 2 days:

You can view requests, responses and emojis statistics
 - for the `/analyze` endpoint at http://localhost:8080/metrics/analyze
 - for the `/analyze-staged` endpoint at http://localhost:8080/metrics/analyze-staged.
 - for the requests coming from gitlab: http://localhost:8080/metrics/analyze-gitlab.

You can retrieve single svg images at the following endpoints:
 - http://localhost:8080/metrics/analyze/requests
 - http://localhost:8080/metrics/analyze/responses
 - http://localhost:8080/metrics/analyze-staged/requests
 - http://localhost:8080/metrics/analyze-staged/responses
 - http://localhost:8080/metrics/analyze-gitlab/requests
 - http://localhost:8080/metrics/analyze-gitlab/responses
 - http://localhost:8080/metrics/analyze-gitlab/emojis

Examples:

```
http GET "localhost:8080/metrics/analyze/requests" > /tmp/plot.svg
curl "localhost:8080/metrics/analyze/staged/requests" > /tmp/plot.svg
```

You can specify the time period in hours, days, or weeks.
The time period:
 - cannot be less than one hour
 - cannot be negative
 - ends at the current time (when the query is made)
 - starts at the specified time interval before the current time.

Examples:

```
http GET "localhost:8080/metrics/analyze/requests?hours=5" > /tmp/plot_hours.svg
http GET "localhost:8080/metrics/analyze/requests?days=5" > /tmp/plot_days.svg
http GET "localhost:8080/metrics/analyze/requests?weeks=5" > /tmp/plot_weeks.svg
```

System Prompts
--------------

Prompt templates used by Log Detective are stored in the `prompts.yml` file.
It is possible to modify the file in place, or provide your own.
In CLI you can override prompt templates location using `--prompts` option,
while in the container service deployment the `LOGDETECTIVE_PROMPTS` environment variable
is used instead.

Prompts need to have a form compatible with python [format string syntax](https://docs.python.org/3/library/string.html#format-string-syntax)
with spaces, or replacement fields marked with curly braces, `{}` left for insertion of snippets.

Number of replacement fields in new prompts, must be the same as in originals.
Although their position may be different.

License
-------

This project is licensed under the Apache-2.0 License - see the LICENSE file for details.
