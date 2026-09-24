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

Submit a log for analysis:

```sh
curl --header "Content-Type: application/json" --request POST \
     --data '{"files":[{"name":"build.log","content":"ERROR: missing dependency"}]}' \
     http://localhost:8080/analyze
```

The request returns `202 Accepted` with a `Location` header for polling.
See the [API guide](docs/api.md) for task status, authentication, Koji and GitLab
integration, metrics, and a completed analysis example. Endpoint schemas are
available from the running server at `/docs` and `/openapi.json`.

## Generate a new database revision with alembic

Modify the database models (`logdetective/database/models/`).

Generate a new database revision with the command:

**Warning**: this command builds the migration image and starts PostgreSQL.

```sh
CHANGE="A change comment" make alembic-generate-revision
```

Procrastinate owns its `procrastinate_*` tables; Alembic owns Log Detective's
tables. On a new database, the one-shot `migrate` service installs the pinned
Procrastinate 3.9 schema and then runs `alembic upgrade head`. Web and worker
replicas must not run schema installation themselves. Before changing the
pinned Procrastinate minor version, operators must apply its supplied SQL
migrations according to that release's notes; the 3.9 schema installer is not
an idempotent upgrade command.

The migration service uses the Fedora-based `quay.io/logdetective/migrate`
image built from `Containerfile.migrate`. It installs the complete Log Detective
wheel without its general application dependencies, then adds only PostgreSQL
client tools and the dependencies in the standard
`[dependency-groups].migration` group. Poetry installs that group at the exact
versions recorded in `poetry.lock`; update the group and lock together whenever
database, model-import, or Procrastinate dependencies change.

This release is a direct queue cut-over. Its Alembic revision refuses to replace
or restore `task_analysis` when that table contains records, preventing silent
loss of accepted work. Rollback therefore requires draining/removing all new
tasks, downgrading Alembic, and only then removing Procrastinate's schema if it
is no longer needed.

Long-running analysis is performed by the `worker` service. The
`maintenance-worker` service consumes only reconciliation and expiry jobs with
one reserved worker slot, so queued or running analyses cannot occupy its
capacity. Deploy both workers; `make server-up` starts both in development.
Outside Compose, start `python -m logdetective.worker analysis` and
`python -m logdetective.worker maintenance` as separate processes.
The API service validates and admits work but does not invoke inference or
outbound GitLab clients. All services load `server/config.yml`; inference
provider settings and per-instance GitLab API tokens are configured there.
Analysis worker concurrency, polling hints, graceful shutdown, stalled-worker
detection, and the default 30-day result retention are configured under
`task_queue` in `server/config.yml`.
The total timeout for downloading a remote log source is configured with
`general.log_source_request_timeout` in that file.
Analysis runs directly in Procrastinate's async worker tasks. Cancellation is
cooperative: async work stops at cancellation points, while an already-running
blocking Koji, GitLab, extraction, sanitization, or embedding call is allowed to
finish before the public task becomes `cancelled`. These blocking calls use
bounded client timeouts where network access is involved. The container's
`stop_grace_period` must remain longer than `shutdown_graceful_timeout`.

The periodic reconciler compares application state with queue job status. It
confirms a `CANCELLING` task when its job is missing, succeeded, aborted, or
cancelled. For other active tasks, a succeeded job without a published outcome
reports `result_lost`. Stalled jobs and other missing or unexpectedly terminal
combinations fail the application task and fence it against late publication.

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
    - Set the related `inference.provider_settings` values in the same file.
4. Mount the Vertex credentials JSON at the path configured by
   `inference.provider_settings.vertex_credentials`.

The server and worker intentionally use the same application configuration and
are therefore in the same configuration trust domain. The separate API bearer
token file selected by `LOGDETECTIVE_TOKENS_FILE` remains server-specific.
Set `general.sentry_dsn` in `server/config.yml` to enable Sentry in both the
API server and worker containers. When it is unset, Sentry is disabled in both.

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

When annotated-snippet lookup is enabled, its embedding model is loaded lazily on
the first analysis that finds annotations to search and is then reused for the life
of that process. Importing Log Detective modules does not load the model.

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
