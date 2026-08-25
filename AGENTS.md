# Log Detective

An LLM-powered build log analyzer for Fedora/RHEL ecosystems. Operates in two modes:
- CLI tool using local llama.cpp inference - `logdetective.logdetective:main`
- FastAPI server launched via gunicorn using BeeAI agent framework with tool-calling LLMs via LiteLLM - `logdetective.server.server`

Container images are published to `quay.io/logdetective/` after each release.
Production uses either vLLM with GPU inference, or Gemini / VertexAI.
Base/dev uses llama.cpp.

# Prerequisites

- Environment variables are in `env_file`; server config is in `server/config.yml`.
- If running a local model, place a GGUF model file in `./models/` (path referenced by `LLAMA_ARG_MODEL` in `env_file`).
- Local model needs a Jinja2 chat template at `./models/chat_template.jinja` for tool-call support.
- If running a model from some provider, obtain appropriate credentials and set `provider_settings` in `server/config.yml`.

# Setup

- Supports python >=3.11,<3.14.
- Poetry manages dependencies (defined in `pyproject.toml`).
- Tox orchestrates test environments (tox base_python is 3.13.).

To install full superset of dependencies in a single resolution pass, use:
`poetry install --extras "server server-testing testing"`

Tox environments use two separate `poetry install` calls.
Combined form is more stable for interactive development.
Dev stack uses `docker-compose-dev.yaml` which extends the base `docker-compose.yaml`.
DB migrations run automatically on server startup via `scripts/await_psql` + alembic.
For CUDA GPU acceleration, uncomment the device lines in `docker-compose-dev.yaml`.

- `make server-up` builds and starts the dev stack (inference, server, postgres, nginx)
- `make server-down` tears down dev stack
- `make rebuild-server` rebuild server image without cache

# Testing

- Core/CLI tests: `tox -e pytest_base`
- Server tests: `tox -e pytest_server` - require podman; run on Postgres + pgvector (see `Container.database`)
- CI runs on GitHub Actions which run tox: pytest (base + server), and all linters (`tox -e lint,style,ruff,djlint`)

# Data modeling conventions

- Pydantic v2 for all request/response validation and config models (`BaseModel`, `Field`, `model_validator`, `field_validator`, `ConfigDict`)
- SQLAlchemy 2.x async ORM with asyncpg driver for database models
- Alembic for DB migrations; autogenerate new ones with `CHANGE="description" make alembic-generate-revision`

# Formatting conventions

- Logging via `logging.getLogger("logdetective")`, initialized in `logdetective/__init__.py` for CLI tool, or `LOG` constant initialized via `get_log()` in `logdetective/server/config.py` for server (using options in `server/config.yml`).
- Linting enforced by: Pylint config in `pyproject.toml` and `.pylintrc.tests`, flake8 and ruff config in `tox.ini`
- Pre-commit hooks: trailing-whitespace, end-of-file-fixer, check-yaml, check-added-large-files, flake8

# Test conventions

- Split into `tests/base` (utilities used in both CLI and server) and `tests/server`.
- Async tests use `@pytest.mark.asyncio` decorator
- Mocking: `unittest.mock` for object mocking/patching, `aioresponses` for async HTTP
- Some test data fixtures (related to gitlab) live in `tests/server/data/` as YAML files

# Package layout

- `logdetective/` - Core: CLI entry point, extractors, prompt management, utilities
- `logdetective/prompts/` - Prompt templates for logdetective
- `logdetective/server/` - FastAPI app, config, routes, GitLab/Koji integrations
- `logdetective/server/agent/` - BeeAI agent and tool definitions (Drain, csgrep, traceback, snippet analysis)
- `logdetective/server/database/` - SQLAlchemy async engine, session factory, transaction helpers
- `logdetective/server/database/models/` - ORM models (metrics, merge requests, koji, annotated builds with pgvector)
- `logdetective/server/templates/` - Jinja2 response templates (HTML, GitLab markdown)
- `alembic/versions/` - Database migration scripts
- `server/` - Deployment configs (gunicorn, nginx templates, server config YAML)
- `tests/base/` - Tests for core package (no DB)
- `tests/server/` - Tests for server package (requires PostgreSQL)

# Documentation

When making functionality changes, check whether these need updating:

- `AGENTS.md` - this file
- `THREAT_MODEL.md` - security assets, entry points, threats
- `README.md` - general usage, installation, configuration overview
- `logdetective.1.asciidoc` - man page; covers all CLI flags defined in `logdetective/logdetective.py:setup_args()`; rebuild with `tox -e manpage`
- `logdetective/logdetective.py:setup_args()` - argparse help strings shown by `logdetective --help`; keep in sync with the man page
- `alembic/er_diagram.md` - Mermaid ER diagram; regenerate with `make generate-db-diagram` after schema changes (alembic revisions)
