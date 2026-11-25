# Database

## Generating a new database revision

After modifying the database models (`logdetective/database/models/`) it's necessary
to generate a new database revision.

**Warning**: this command builds the migration image and starts PostgreSQL.

```sh
CHANGE="A change comment" make alembic-generate-revision
```

Then, running `make rebuild-migrate` so that the migration image contains the new alembic revision scripts.
Only then will `make dev-up` properly apply migrations for the dev stack.

## Task management

Information about long running tasks handled by [Procrastinate](https://github.com/procrastinate-org/procrastinate) is stored in its `procrastinate_*` tables.

Alembic owns only Log Detective's tables. See the [database ER diagram](../alembic/er_diagram.md)
for the schema and [worker reconciliation](workers.md#reconciliation) for how the worker
compares application task state with queue job status.

On a new database, the one-shot `migrate` service installs the pinned
Procrastinate 3.9 schema and then runs `alembic upgrade head`.

Before changing the pinned Procrastinate minor version, operators must apply its supplied SQL migrations according to that release's notes; the 3.9 schema installer is not an idempotent upgrade command.

The migration service uses the Fedora-based `quay.io/logdetective/migrate`
image built from `../containers/Containerfile.migrate`.

The image contains only mininimum required dependencies from the `[dependency-groups].migration` group.
