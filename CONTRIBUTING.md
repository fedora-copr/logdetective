# Contributing

Contributions are welcome! Submit a pull request if you have improvements or new features to add. Make sure your changes pass all existing tests before submitting.
For larger code changes, please create an issue first.

We are still looking for more annotated snippets that will increase the quality of Log Detective's results. You can contribute on our [website](https://logdetective.com/).

Use pre-commit to ensure that your code meets basic linting requirements.

## Tests

Tests must placed in the `./tests/` path.

Postgresql client utilities are required to run tests.
```sh
dnf install postgresql
```

The [tox](https://github.com/tox-dev/tox) is used to manage tests.

```sh
tox
```
Will create a virtual environment with dependencies and run all the tests. For more information follow the tox help.

To run only tests:

```sh
tox run -e pytest
```

## VS Code testing with podman/docker-compose

- In `containers/Containerfile`, add `debugpy` as a dependency

```diff
+RUN pip3 install debugpy
```

- Rebuild server image with new dependencies

```sh
make rebuild-server
```

- Forward debugging port in `containers/docker-compose.yaml` for `server` service.

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
cd containers
podman-compose --project-name logdetective -f docker-compose-dev.yaml up server
```

- Run Visual Stdio Code debug configuration named *Python Debug: Remote Attach*
