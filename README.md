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

We provide two example deployment configurations. The [development configuration](./containers/docker-compose-dev.yaml) is intended for local testing of changes, and uses own [llama.cpp server image](https://github.com/ggml-org/llama.cpp/pkgs/container/llama.cpp).

The sample [production](./containers/docker-compose-prod.yaml) configuration, uses 4 load balanced [vLLM](https://github.com/vllm-project/vllm) servers.

Use the make targets from the project root directory to start and stop either deployment. The development and production files extend `containers/docker-compose.yaml`. The installed `podman-compose` mishandles their relative bind mounts when invoked with a `containers/...` file path. The Make targets avoid this by running Compose from `containers/`.

The basic setup:

1. Make sure your `MODELS_PATH` environment variable points to a directory with your local LLM files.
You can either edit the value in [env_file](env_file), create a symlink `ln -s /directory/with/your/llms ./models`, or:
    ```sh
    $ export MODELS_PATH=/path/to/models/
    $ ll $MODELS_PATH
    -rw-r--r--. 1 tt tt 3.9G apr 10 17:18  granite-4.0-h-tiny-Q8_0.gguf
    ```
2. Start the development stack with `make dev-up`, or the production stack with `make prod-up`. Both run in the background. Stop them with `make dev-down` or `make prod-down`.
   For direct Podman Compose commands, change to `containers/` first and use the Compose filename there, see make targets.
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
Endpoint schemas are available from the running server at `/docs` and `/openapi.json`.

## Documentation

- [API guide](docs/api.md): API structure and operation
- [Agents](docs/agents.md): general and Testing Farm analysis workflows
- [Tools](docs/tools.md): tool operation
- [Workers](docs/workers.md): worker architecture and task processing
- [Database](docs/database.md): generating migrations and table ownership

## Production instance

Our FastAPI server and model inference server run through `podman-compose` on an
Amazon AWS instance. The VM is provisioned by an
[ansible playbook](https://pagure.io/fedora-infra/ansible/blob/main/f/roles/logdetective/tasks/main.yml).

You can control the server through:

```sh
cd /root/logdetective/containers
podman-compose --project-name logdetective -f docker-compose-prod.yaml ...
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


## Choice of LLM

The build agent's extractor, database lookup, and snippet annotation workflow is described
in [Build analysis agent](docs/agents.md#build-analysis-agent).

Log Detective requires an instruction tuned model with tool calling to function properly.

Whether or not the model has been trained to work with instructions can be determined by examining the model card, or simply by checking if it has `instruct` in its name.

Log Detective uses `/chat/completions` API as defined by OpenAI. The API must support
both `system` and `user` roles, in order to properly work with a system prompt.
The `system` role defaults to `developer`

Configuration field `system_role` can be used to set role name for APIs with non-standard roles.
However, proper function of Log Detective can not be guaranteed in such cases.

## License

This project is licensed under the `Apache-2.0 License`. See the [LICENSE](./LICENSE) file for details.
