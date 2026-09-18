# Log Detective

An LLM-powered build log analyzer for Fedora/RHEL ecosystems.
Operates as a FastAPI server launched via gunicorn.
Uses BeeAI agent framework with tool-calling LLMs via LiteLLM - `logdetective.server`

Container images are published to `quay.io/logdetective/` after each release.
Production uses either vLLM with GPU inference, or Gemini / VertexAI.

# Prerequisites

- Environment variables are in `env_file`; server config is in `server/config.yml`.
- If running a local model, place a GGUF model file in `./models/` (path referenced by `LLAMA_ARG_MODEL` in `env_file`).
- Local model needs a Jinja2 chat template at `./models/chat_template.jinja` for tool-call support.
- If running a model from a provider, set its credentials under `inference.provider_settings` in `server/config.yml`.

# Setup

- Supports python >=3.11,<3.14.
- Poetry manages dependencies (defined in `pyproject.toml`).
- Tox orchestrates test environments (tox base_python is 3.13.).

To install full superset of dependencies in a single resolution pass, use:
`poetry install --extras "testing"`

Tox environments use two separate `poetry install` calls.
Combined form is more stable for interactive development.
Dev stack uses `docker-compose-dev.yaml` which extends the base `docker-compose.yaml`.
On a new database, the one-shot `migrate` service installs Procrastinate's pinned schema and then runs Alembic before API and worker services start. Procrastinate upgrades require its supplied SQL migrations; its 3.9 schema installer is not an idempotent upgrader.
The migration service installs the complete project wheel without dependencies in the minimal Fedora-based `Containerfile.migrate`. Its runtime subset is declared in the standard `[dependency-groups].migration` group and installed at the versions recorded in `poetry.lock`.
For CUDA GPU acceleration, uncomment the device lines in `docker-compose-dev.yaml`.

- `make server-up` builds and starts the dev stack (inference, server, analysis worker, maintenance worker, postgres, nginx)
- `make server-down` tears down dev stack
- `make rebuild-server` rebuild server image without cache

# Testing

- `tox -e pytest` - requires podman; runs on Postgres + pgvector (see `Container.database`)
- CI runs on GitHub Actions which run `tox -e pytest` + `tox -e lint,style,ruff,djlint`

# Data modeling conventions

- Pydantic v2 for all request/response validation and config models (`BaseModel`, `Field`, `model_validator`, `field_validator`, `ConfigDict`)
- SQLAlchemy 2.x async ORM with the psycopg 3 driver for database models
- Alembic for DB migrations; autogenerate new ones with `CHANGE="description" make alembic-generate-revision`
- Procrastinate owns only its `procrastinate_*` schema; do not copy its schema into Alembic revisions.

# Asynchronous API convention

- Every endpoint that starts non-trivial or long-running work must durably admit a Procrastinate job and its application record in one database transaction, then return `202` only after commit.
- Return a stable task envelope and `Location` plus `Retry-After` headers. Polling uses `GET /tasks/{opaque_uuid}` and returns `200` in both active and terminal states.
- Cancellation uses `DELETE /tasks/{opaque_uuid}` and application state is the durable cancellation authority. Never expose Procrastinate job IDs or tables through the public API.
- Client UUIDs are idempotency keys. Identical retries return the same unexpired operation; conflicting or expired reuse returns `409` until the expired record is removed.
- Keep inputs, results, ownership, public state, retention, and fencing in the application model. Procrastinate owns scheduling, claiming, and worker heartbeats.
- Analysis work runs directly in Procrastinate's async worker tasks. Blocking library calls are tracked and cancellation remains pending until they return. The API tier shares application configuration but must not invoke inference or outbound GitLab clients.
- Run separate Procrastinate workers for `analysis` and `maintenance`; the maintenance worker has one reserved job slot, while `task_queue.concurrency` applies to the analysis worker. Both are required in deployments so analysis jobs cannot occupy maintenance capacity.
- Queue reconciliation treats application state and Procrastinate job status separately. `mark_queue_failure()` fences stalled jobs and active tasks whose jobs disappear or terminate unexpectedly outside cancellation confirmation. A succeeded job with a non-cancelling active task and no application outcome uses `mark_result_lost()`.
- GitLab's external webhook remains `204`, but acknowledgement occurs only after durable queue admission. Koji uses the same task resource as generic analysis and has no callback endpoint.
- Database operations belong on relevant model classmethods, must be type annotated, and every nullable database result must be checked explicitly.

# Formatting conventions

- Logging via `logging.getLogger("logdetective")`, initialized in `logdetective/__init__.py` or `LOG` constant initialized via `get_log()` in `logdetective/config.py` for server (using options in `server/config.yml`).
- Linting enforced by: Pylint config in `pyproject.toml` and `.pylintrc.tests`, flake8 and ruff config in `tox.ini`
- Pre-commit hooks: trailing-whitespace, end-of-file-fixer, check-yaml, check-added-large-files, flake8

# Test conventions

- Async tests use `@pytest.mark.asyncio` decorator
- Mocking: `unittest.mock` or `flexmock` for object mocking/patching, `aioresponses` for async HTTP
- Some test data fixtures (related to gitlab) live in `tests/data/` as YAML files

# Package layout

- `logdetective/` - FastAPI app, Procrastinate worker/tasks, config, routes, GitLab/Koji integrations, extractors, prompt management, utilities
- `logdetective/prompts/` - Prompt templates for logdetective
- `logdetective/agent/` - BeeAI agent and tool definitions (Drain, csgrep, traceback, snippet analysis)
- `logdetective/database/` - SQLAlchemy async engine, session factory, transaction helpers
- `logdetective/database/models/` - ORM models (metrics, merge requests, koji, annotated builds with pgvector)
- `logdetective/templates/` - Jinja2 response templates (HTML, GitLab markdown)
- `alembic/versions/` - Database migration scripts
- `server/` - Deployment configs (gunicorn, nginx templates, server config YAML)
- `tests/` - Tests for utilities and server (requires PostgreSQL)

# Documentation

When making functionality changes, check whether these need updating:

- `AGENTS.md` - this file
- `THREAT_MODEL.md` - security assets, entry points, threats
- `README.md` - general usage, installation, configuration overview
- `alembic/er_diagram.md` - Mermaid ER diagram; regenerate with `make generate-db-diagram` after schema changes (alembic revisions)
