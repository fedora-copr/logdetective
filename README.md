# Log Detective

[![PyPI - Version](https://img.shields.io/pypi/v/logdetective?color=blue)][PyPI Releases]

[PyPI Releases]: https://pypi.org/project/logdetective/#history

A tool, service and RHEL process integration to analyze logs using Large Language Model (LLM) and a [Drain template miner](https://github.com/jpodivin/Drain3-improved) within [BeeAI agentic framework](https://github.com/i-am-bee/beeai-framework).

Service explaining logs is available at: https://logdetective.com/explain

*Note: code of the logdetective.com website is at [github.com/fedora-copr/logdetective-website](https://github.com/fedora-copr/logdetective-website).*

Note: Log Detective used to be developed as both a CLI tool and a FastAPI server.
The CLI tool is now deprecated and has been removed from this repository since 5.0 release.
Please keep in mind that there still might be traces and references to it.


# Server

For locally setting up the FastAPI server, you would need a postgresql and some inference server.

Log Detective has been built as inference agnostic service. The only requirement, is that the inference server must provide OpenAI API.

We provide two example deployment configurations. The [development configuration](./docker-compose-dev.yaml) is intended for local testing of changes, and uses own [llama.cpp server image](https://github.com/ggml-org/llama.cpp/pkgs/container/llama.cpp).

The sample [production](./docker-compose-prod.yaml) configuration, uses 4 load balanced [vLLM](https://github.com/vllm-project/vllm) servers.

The basic setup:

1. Make sure your `MODELS_PATH` environment variable points to a directory with your local LLM files.
You can either edit the value in [env_file](env_file), create a symlink `ln -s /directory/with/your/llms ./models`, or:
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

Modify the database models (`logdetective/database/models/`).

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

## Using with Vertex AI

To be able to use Log Detective with Vertex AI:
1. You will need to have access to the Service account ADC (Application Default Credentials) JSON file
    - To use our (Log Detective project) Vertex AI Service account credentials, you will need access to our Bitwarden vault.
    - Alternatively, you can use Google Cloud Platform in order to generate a new credential file.
2. Put the credentials JSON file into the project directory as `log-detective-vertex.json`. Without this file, container creation will fail.
3. Update `server/config.yml`:
    - Change `inference.model` to `vertexai:model-name`, such that `model-name` is a valid model provided by Vertex AI.
    - Set the additional related config values in `server/config.yml` (follow the provided instructions, everything is set up so that you can just uncomment the 3 `GOOGLE_`* values).
4. Uncomment the line in `docker-compose.yaml` which mounts the credentials JSON file.

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

Prompts are defined as Jinja templates and placed in location specified `LOGDETECTIVE_PROMPT_TEMPLATES` (`logdetective/prompts` by default) environment variable of the container service.
It is possible to add extra sources/references for the agent via `server/config.yml` file (`prompts` section, under `references`).

All system prompt templates must include place for `system_time` variable.

If `references` list is defined in `server/config.yml`, templates must also include a handling for a list of references.

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
Users can specify regular expressions matching such chunks using the Skip Snippets feature.

Patterns are defined in a TOML file. Each entry is a TOML table with a required `pattern` key
(a regular expression) and an optional `files` key listing exact filenames the pattern applies to.
When `files` is omitted the pattern applies to every log file processed.

Use single-quoted TOML strings for patterns — they are taken verbatim with no escape processing,
so backslashes and other special characters work as-is.

```toml
# applies to every file
[child_exit_code_zero]
pattern = '.*Child return code was: 0'

# applies only to backend.log and app.log
[skip_debug_messages]
pattern = '^DEBUG:.*'
files = ['backend.log', 'app.log']
```

Example of a valid pattern definition file: `logdetective/skip_snippets.toml`,
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

While in server mode, the `csgrep` field in `extractor` config needs to be set to `true`.

```yaml
csgrep: true
```

Both options are disabled by default and error will be produced if the option is used,
but `csgrep` is not present in the $PATH.

The container images are built with `csdiff` installed.

## Real Example

Log Detective can work with any logs, though we optimize it for RPM build logs.
The following output is a response for the `/analyze` endpoint.

The analyzed build: https://koji.fedoraproject.org/koji/taskinfo?taskID=149750933

You can get similar output by running this on your local compose:

```sh
curl --header "Content-Type: application/json" --request POST \
     --data '{
        "files": [
            {
                "name": "root.log",
                "url": "https://kojipkgs.fedoraproject.org//work/tasks/933/149750933/root.log"
            },
            {
                "name": "mock_output.log",
                "content": "https://kojipkgs.fedoraproject.org//work/tasks/933/149750933/mock_output.log"
            },
            {
                "name": "build.log",
                "content": "https://kojipkgs.fedoraproject.org//work/tasks/933/149750933/build.log"
            }
        ],
        "build_metadata": {
            "specfile": null,
            "last_patch": null,
            "commentary": "Logs are from a Koji build.\nKoji builds use mock chroots; build.log contains build output,\nroot.log has dependency resolution and mock setup.",
            "infra_status": null
        }
     }' \
     http://localhost:8080/analyze
```

Note that only a handful of snippets were selected from the original response for demonstration purposes:

```json
{
  "explanation": {
    "text": "The build failed during the compilation phase of `emacs-with-editor` because the build process could not find a required load file, specifically `cond-let`, while compiling `with-editor.el` (build.log, line 104). This indicates a missing dependency or an incomplete build environment setup for Emacs Lisp components."
  },
  "no_issue_found": false,
  "snippets": [
    {
      "line_number": 585,
      "source_file": "root.log",
      "text": "DEBUG util.py:535:  Warning: skipped OpenPGP checks for 124 packages from repository: build"
    },
    {
      "line_number": 5276,
      "source_file": "root.log",
      "text": "DEBUG util.py:535:  Package \"emacs-1:31.1-2.fc46.ppc64le\" is already installed."
    },
    {
      "line_number": 3,
      "source_file": "mock_output.log",
      "text": "INFO: mock.py version 6.8 starting (python version = 3.14.7, NVR = mock-6.8-1.fc44), args: /usr/libexec/mock/mock -r koji/f46-build-70856970-6688917 --new-chroot --init"
    },
    {
      "line_number": 275,
      "source_file": "mock_output.log",
      "text": "INFO: Buildroot is handled by package management installed into bootstrap:\n  rpm-6.1.0-1.fc46.ppc64le\n  rpm-sequoia-1.10.2-5.fc45.ppc64le\n  dnf5-5.4.3.0-2.fc46.ppc64le\n  dnf5-plugins-5.4.3.0-2.fc46.ppc64le"
    },
    {
      "line_number": 287,
      "source_file": "mock_output.log",
      "text": "ERROR: Exception(/var/tmp/koji/tasks/933/149750933/local/work/tasks/907/149750907/emacs-with-editor-3.5.4-1.fc46.src.rpm) Config(f46-build-70856970-6688917) 3 minutes 59 seconds"
    },
    {
      "line_number": 289,
      "source_file": "mock_output.log",
      "text": "ERROR: Command failed: \n # /usr/bin/systemd-nspawn -q -M ee22ca6a19f040769a6c490430569274 -D /var/lib/mock/f46-build-70856970-6688917/root -a -u mockbuild --capability=cap_ipc_lock --capability=cap_mac_admin --bind=/dev/btrfs-control --bind=/dev/mapper/control --bind=/dev/fuse --bind=/dev/loop-control --bind=/dev/loop0 --bind=/dev/loop1 --bind=/dev/loop2 --bind=/dev/loop3 --bind=/dev/loop4 --bind=/dev/loop5 --bind=/dev/loop6 --bind=/dev/loop7 --bind=/dev/loop8 --bind=/dev/loop9 --bind=/dev/loop10 --bind=/dev/loop11 --resolv-conf=off --console=pipe --setenv=TERM=vt100 --setenv=SHELL=/bin/bash --setenv=HOME=/builddir --setenv=HOSTNAME=mock --setenv=PATH=/usr/bin:/bin:/usr/sbin:/sbin '--setenv=PROMPT_COMMAND=printf \"\\033]0;<mock-chroot>\\007\"' '--setenv=PS1=<mock-chroot> \\s-\\v\\$ ' --setenv=LANG=C.UTF-8 bash --login -c '/usr/bin/rpmbuild -bb --noclean --target noarch --nodeps /builddir/build/SPECS/emacs-with-editor.spec'"
    },
    {
      "line_number": 118,
      "source_file": "build.log",
      "text": "Cannot find a locale compatible with document strings translations"
    },
    {
      "line_number": 133,
      "source_file": "build.log",
      "text": "RPM build errors:"
    },
    {
      "line_number": 134,
      "source_file": "build.log",
      "text": "error: Bad exit status from /var/tmp/rpm-tmp.Us2T6p (%build)\n    Bad exit status from /var/tmp/rpm-tmp.Us2T6p (%build)"
    }
  ],
  "solution": {
    "text": "Ensure that all necessary Emacs Lisp development dependencies, including any required libraries that provide `cond-let`, are correctly installed and available in the build environment before running the build process."
  }
}
```

The most significant field for diagnosis is `explanation`.

## Choice of LLM

While Log Detective is compatible with a wide range of LLMs, it does require an instruction tuned model with tool calling to function properly.

Whether or not the model has been trained to work with instructions can be determined by examining the model card, or simply by checking if it has `instruct` in its name.

When deployed as a server, Log Detective uses `/chat/completions` API as defined by OpenAI. The API must support both `system` and `user` roles, in order to properly work with a system prompt. The `system` role defaults to `developer`

Configuration field `system_role` can be used to set role name for APIs with non-standard roles.
However, proper function of Log Detective can not be guaranteed in such cases.


# Contributing

Contributions are welcome! Please submit a pull request if you have any improvements or new features to add. Make sure your changes pass all existing tests before submitting.
For larger code changes, please consult us first by creating an issue.

We are always looking for more annotated snippets that will increase the quality of Log Detective's results. You can contribute on our [website](https://logdetective.com/).

Please use pre-commit to ensure that your code meets basic linting requirements.


# Tests

Tests for code (server or utilities) must placed in the `./tests/` path.

The [tox](https://github.com/tox-dev/tox) is used to manage tests. Please install `tox` package into your distribution and run:
```sh
tox
```
This will create a virtual environment with dependencies and run all the tests. For more information follow the tox help.

To run tests in the tox environment:

```sh
tox run -e pytest
```

To run the test suite, you will need postgresql client utilities.
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

## License

This project is licensed under the `Apache-2.0 License`. See the [LICENSE](./LICENSE) file for details.
