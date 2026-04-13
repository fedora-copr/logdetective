# Log Detective

[![PyPI - Version](https://img.shields.io/pypi/v/logdetective?color=blue)][PyPI Releases]

[PyPI Releases]: https://pypi.org/project/logdetective/#history

A tool, service and RHEL process integration to analyze logs using a Large Language Model (LLM) and a [Drain template miner](https://github.com/logpai/Drain3).

Service explaining logs is available at: https://logdetective.com/explain

Note: if you are looking for code of website logdetective.com it is in [github.com/fedora-copr/logdetective-website](https://github.com/fedora-copr/logdetective-website).


# Server

For locally setting up the FastAPI server, you would need a postgresql and some inference server.

Log Detective has been built as inference agnostic service. The only requirement, is that the inference server must provide OpenAI API.

We provide two example deployment configurations. The [development configuration](./docker-compose-dev.yaml) is intended for local testing of changes, and uses our own [llama.cpp server image](https://quay.io/repository/logdetective/inference).

The sample [production](./docker-compose-prod.yaml) configuration, uses 4 load balanced [vLLM](https://github.com/vllm-project/vllm) servers.

The basic setup:

1. Make sure your `MODELS_PATH` environment variable points to a directory with your local LLM files.
You can either edit the value in [.env](.env), create a symlink `ln -s /directory/with/your/llms ./models`, or:
    ```sh
    $ export MODELS_PATH=/path/to/models/
    $ ll $MODELS_PATH
    -rw-r--r--. 1 tt tt 3.9G apr 10 17:18  granite-4.0-h-tiny-Q8_0.gguf
    ```
2. `podman-compose  -f <you-compose-file> up` (or  `podman-compose  -f <you-compose-file> up -d` to detach from your current terminal)
3. When encountering timeout errors (you can check what happens in containers with `podman logs`), If you get `nginx` timeouts, try setting/increasing timeouts in [server/nginx_dev.conf.template](server/nginx_dev.conf.template):
    ```diff
        server {
        listen ${INFERENCE_PROXY_PORT};
    +   proxy_connect_timeout 300s;
    +   proxy_send_timeout 300s;
    +   proxy_read_timeout 300s;
        location / {
            proxy_pass http://inference_backend;
            proxy_set_header Host $host;
        }
    ```

If the `MODELS_PATH` variable is not set, `./models` is mounted inside by default.

Models can be downloaded from [our Hugging Space](https://huggingface.co/fedora-copr).

## Usage

API allows for submission of multiple build artifacts for analysis.
These can be provide using URL, or as raw strings.

```sh
curl --header "Content-Type: application/json" --request POST \
     --data '{
          "files": [
            {
                "name": "build.log",
                "url": "https://url.to/build.log"
            },
            {
                "name": "raw_string.log",
                "content": "Raw string that will be analyzed."
            }
        ],
        "build_metadata": {
            "specfile": null,
            "last_patch": null,
            "commentary": "BuildError: error building package (arch noarch), mock exited with status 30; see root.log for more information",
            "infra_status": null
        }
     }' \
     http://localhost:8080/analyze
```

Note that Log Detective redacts certain personal information, such as emails and GPG fingerprints from logs, before calling LLM.

LLM should be aware of this fact and factor it into its responses.

## Generate a new database revision with alembic

Modify the database models (`logdetective/server/database/models/`).

Generate a new database revision with the command:

**Warning**: this command will start up a new server
and shut it down when the operation completes.

```sh
CHANGE="A change comment" make alembic-generate-revision
```

## Our production instance

Our FastAPI server and model inference server run through `podman-compose` on an
Amazon AWS instance. The VM is provisioned by an
[ansible playbook](https://pagure.io/fedora-infra/ansible/blob/main/f/roles/logdetective/tasks/main.yml).

You can control the server through:

```sh
cd /root/logdetective
podman-compose -f docker-compose-prod.yaml ...
```

The `/root` directory contains valuable data. If moving to a new instance,
please backup the whole directory and transfer it to the new instance.

In order to run containers with Nvidia GPU support, you need to have generate a CDI specification, which can be done through:

```sh
nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

HTTPS certificate generated through:

```sh
certbot certonly --standalone -d logdetective01.fedorainfracloud.org
```

Certificates need to be be placed into location specified by the `LOGDETECTIVE_CERTDIR`
env var and the service should be restarted.

## Querying statistics

You can query requests, responses and emojis statistics via `metrics` endpoints.
They return JSON data with `time_series` array containing metric objects with `metric`, `timestamps`, and `values` fields.
Metrics are `GET` methods and have the form `/metrics/ENDPOINT_TYPE/QUERY_TYPE?parameter=value`:

1. `ENDPOINT_TYPE`: `analyze`, or `analyze-gitlab`.

2. `QUERY_TYPE`:
- `requests` will return how many requests did the server receive at given endpoint.
- `responses` will return average response times during the time period.
- `emojis` will return ALL emoji reactions. This data is collected only for `analyze-gitlab` events, so the `ENDPOINT_TYPE` in the URL is ignored when querying for emojis.
- `all` will retrieve all of the above. If `QUERY_TYPE` is left empty, it defaults to `all`.

3. `parameter=value` will specify the latest period for which metrics are returned. If unspecified, the query defaults to the last 2 days.
- `parameter` is either `hours`, `days`, `weeks`.
- `value` is a positive integer.
- `parameter` type also controls the granularity of the response: `?days=2` will produce time series with max 2 entries, `?hours=48` will produce a time series with max 48 entries.


Examples:
```sh
curl "http://localhost:8080/metrics/analyze-gitlab/emojis?days=5"
```

## System Prompts

Prompts are defined as Jinja templates and placed in location specified by `--prompt-templates` option of the CLI utility, or `LOGDETECTIVE_PROMPT_TEMPLATES` environment variable of the container service. With further, optional, configuration in the `prompts.yml` configuration file.

All system prompt templates must include place for `system_time` variable.

If `references` list is defined in `prompts.yml`, templates must also include a handling for a list of references.

Example:

```jinja
{% if references %}
## References:

    {% for reference in references %}
    * {{ reference.name }} : {{ reference.link }}
    {% endfor %}
{% endif %}

```

## Skip Snippets

Certain log chunks may not contribute to the analysis of the problem under any circumstances.
User can specify regular expressions, matching such log chunks, along with simple description,
using Skip Snippets feature.

Patterns to be skipped must be defined yaml file as a dictionary, where key is a description
and value is a regular expression. For example:

```yaml
child_exit_code_zero: "Child return code was: 0"
```

Special care must be taken not to write a regular expression which may match
too many chunks, or which may be evaluated as data structure by the yaml parser.

Example of a valid pattern definition file: `logdetective/skip_snippets.yml`,
can be used as a starting point and is used as a default if no other definition is provided.


## Extracting snippets with csgrep

When working with logs containing messages from GCC, it can be beneficial to employ
additional extractor based on `csgrep` tool, to ensure that the messages are kept intact.
Since `csgrep` is not available as a python package, it must be installed separately,
with a package manager or from [source](https://github.com/csutils/csdiff).

The binary is available as part of `csdiff` package on Fedora.

```sh
dnf install csdiff
```

When working with CLI Log Detective, the csgrep extractor can be activated using option `--csgrep`.
While in server mode, the `csgrep` field in `extractor` config needs to be set to `true`.

```yaml
csgrep: true
```

Both options are disabled by default and error will be produced if the option is used,
but `csgrep` is not present in the $PATH.

The container images are built with `csdiff` installed.

# Command Line Tool

## Installation

**Fedora 41+**

```sh
dnf install logdetective
```

**From Pypi repository**

The logdetective project is published on the [Pypi repository](https://pypi.org/project/logdetective/). The `pip` tool can be used for installation.

First, ensure that the necessary dependencies for the `llama-cpp-python` project are installed. For Fedora, install `gcc-c++`:

```sh
dnf install gcc-c++
```

Then, install the `logdetective` project using pip:

```sh
pip install logdetective
```

**Local repository install**

Clone this repository and install with pip:

```sh
pip install .
```

## Usage

To analyze a log file, run the script with the following command with:

**Required arguments:**
- `file`: The path or URL of the log file to be analyzed.

**Optional arguments:**
- `-M, --model MODEL_NAME` (default: "fedora-copr/granite-3.2-8b-instruct-GGUF"): The path or Hugging space name of the language model for analysis. For models from Hugging Face, write them as `namespace/repo_name`. As we are using LLama.cpp we want this to be in the `gguf` format. If the model is already on your machine it will skip the download.
- `-F | --filename-suffix SUFFIX` (default `Q4_K.gguf`): You can specify which suffix of the model file to use. This option is applied when specifying model (from the different quantizations) using the Hugging Face repository.
- `-C | --n-clusters N` (default 8): Number of clusters for Drain to organize log chunks into. This only makes sense when you are summarizing with Drain.
- `-n, --no-stream`: Print the full response at once, instead of token-by-token.
- `-v, --verbose`: Increase output verbosity. Can be used multiple times (-v, -vv, -vvv) for different debug levels.
- `-q, --quiet`: Suppress all output except the explanation.
- `--prompts PROMPTS` (DEPRECATED, replaced by `--prompts-config`) Path to prompt configuration file.
- `--prompts-config PROMPTS` (default `logdetective/prompts.yml`): Path to prompt configuration file.
- `--prompt-templates TEMPLATE_DIR` (default `logdetective/prompts`): Path to prompt template directory. Prompts must be valid Jinja templates, and system prompts must include field `system_time`.
- `--temperature NUM` (default `0.0`): Temperature for inference. Higher temperatures lead to more creative, random responses.
- `--skip-snippets SNIPPETS` (default `logdetective/skip_snippets.yml`): Path to patterns for skipping snippets.
- `--csgrep`: Use `csgrep` to process the log. Requires `csgrep` to be installed separately.
- `--mib_limit NUMBER` Limits the size (in MiB) of request (if submitting raw files) or file (if submitting via URL) for analyze endpoints (default 50). Logs or requests exceeding this will be rejected.

**Examples:**

Analyzing a log via URL or stored locally:

```sh
logdetective https://example.com/logs.txt
logdetective ./data/logs.txt
```

Examples of using different models. Note the use of `--filename-suffix` (or `-F`) option, useful for models that were quantized:

```sh
logdetective https://example.com/logs.txt --model QuantFactory/Meta-Llama-3-8B-Instruct-GGUF --filename-suffix Q5_K_S.gguf
logdetective https://kojipkgs.fedoraproject.org//work/tasks/3367/131313367/build.log --model 'fedora-copr/granite-3.2-8b-instruct-GGUF' -F Q4_K_M.gguf
```
Example of altered prompts:

```sh
cp -r ~/.local/lib/python3.13/site-packages/logdetective/prompts ~/my-prompts
vi ~/my-prompts/system_prompt.j2 # edit the system prompt there to better fit your needs
logdetective https://kojipkgs.fedoraproject.org//work/tasks/3367/131313367/build.log --prompt-templates ~/my-prompts
```

Note that streaming with some models (notably Meta-Llama-3) is broken and can be worked around by `no-stream` option:

```sh
logdetective https://example.com/logs.txt --model QuantFactory/Meta-Llama-3-8B-Instruct-GGUF --filename-suffix Q5_K_M.gguf --no-stream
```

## Choice of LLM

While Log Detective is compatible with a wide range of LLMs, it does require an instruction tuned model with tool calling to function properly.

Whether or not the model has been trained to work with instructions can be determined by examining the model card, or simply by checking if it has `instruct` in its name.

When deployed as a server, Log Detective uses `/chat/completions` API as defined by OpenAI. The API must support both `system` and `user` roles, in order to properly work with a system prompt. The `system` role defaults to `developer`

Configuration field `system_role` can be used to set role name for APIs with non-standard roles.
However, proper function of Log Detective can not be guaranteed in such cases.

## Real Example

Log Detective can work with any logs, though we optimize it for RPM build logs.

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

The most significant lines of a logfile wrapped in `[ ] : ` and followed by textual explanation of the log text done by local LLM.


# Contributing

Contributions are welcome! Please submit a pull request if you have any improvements or new features to add. Make sure your changes pass all existing tests before submitting.
For larger code changes, please consult us first by creating an issue.

We are always looking for more annotated snippets that will increase the quality of Log Detective's results. You can contribute on our [website](https://logdetective.com/).

Please use pre-commit to ensure that your code meets basic linting requirements.

# Tests

Tests for code used by server must placed in the `./tests/server/` path, while tests for general
code must be in the `./tests/base/` path.

The [tox](https://github.com/tox-dev/tox) is used to manage tests. Please install `tox` package into your distribution and run:
```sh
tox
```
This will create a virtual environment with dependencies and run all the tests. For more information follow the tox help.

Tox environments for base and server tests are separate, each installs different dependencies. You can run test environments separately, like this:

```sh
tox run -e pytest_base # running base tests:
tox run -e pytest_server # running server tests
```

To run server test suite you will need postgresql client utilities.
```sh
dnf install postgresql
```

## Visual Studio Code testing with podman/docker-compose

- In `Containerfile`, add `debugpy` as a dependency

```diff
+RUN pip3 install debugpy
```

- Rebuild server image with new dependencies

```sh
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

- Prepare `.vscode/launch.json` configuration for Visual Studio Code (at least the following configuration is needed)

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

```sh
podman-compose -f docker-compose-dev.yaml up server
```

- Run Visual Stdio Code debug configuration named *Python Debug: Remote Attach*

## Visual Studio Code CLI debugging

When debugging the CLI application, the `./scripts/debug_runner.py` script can be used
as a stand in for stump script created during package installation.

Using `launch.json`, or similar alternative, arguments can be specified for testing.

Example:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Debug Installed Module",
            "type": "debugpy",
            "request": "launch",
            "console": "integratedTerminal",
            "program": "${workspaceFolder}/scripts/debug_runner.py",
            "args": [<URL_OF_A_LOG>]
        }
    ]
}
```

## License

This project is licensed under the `Apache-2.0 License`. See the [LICENSE](./LICENSE) file for details.
