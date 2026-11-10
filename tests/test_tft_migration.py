"""Test the single Testing Farm schema revision."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import os
import subprocess
import sys
from uuid import uuid4

import pytest
import psycopg
from psycopg import sql


REVISION_PATH = Path(__file__).parent.parent.joinpath(
    "alembic", "versions", "c91b7e46a0d2_add_testing_farm_analysis.py"
)
SPEC = spec_from_file_location("testing_farm_revision", REVISION_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot load Testing Farm migration")
REVISION = module_from_spec(SPEC)
SPEC.loader.exec_module(REVISION)


class FakeBind:  # pylint: disable=too-few-public-methods
    """Return whether the migrated records remain in use."""

    def __init__(self, in_use):
        self.in_use = in_use

    def scalar(self, _statement):
        return self.in_use


class FakeOperations:
    """Capture schema actions without changing a real database."""

    def __init__(self, in_use=False):
        self.bind = FakeBind(in_use)
        self.statements = []

    def get_bind(self):
        return self.bind

    def execute(self, statement):
        self.statements.append(statement)

    def add_column(self, table, column):
        self.statements.append(("add_column", table, column.name))

    def create_index(self, name, table, columns):
        self.statements.append(("create_index", name, table, columns))

    def drop_index(self, name, table_name):
        self.statements.append(("drop_index", name, table_name))

    def drop_column(self, table, column):
        self.statements.append(("drop_column", table, column))


def test_tft_migration_adds_suite_and_metrics_labels(monkeypatch):
    """One revision adds both labels needed by the Testing Farm endpoint."""
    operations = FakeOperations()
    monkeypatch.setattr(REVISION, "op", operations)
    REVISION.upgrade()
    assert operations.statements == [
        "ALTER TYPE tasktype ADD VALUE IF NOT EXISTS 'tft_suite'",
        "ALTER TYPE endpointtype ADD VALUE IF NOT EXISTS 'analyze_tft'",
        ("add_column", "task_analysis", "deadline_at"),
        ("create_index", "ix_task_analysis_deadline_at", "task_analysis", ["deadline_at"]),
    ]


@pytest.mark.parametrize("in_use", [True, None])
def test_tft_migration_refuses_unsafe_downgrade(monkeypatch, in_use):
    """Unknown or retained suite tasks or metrics must remain readable."""
    operations = FakeOperations(in_use)
    monkeypatch.setattr(REVISION, "op", operations)
    with pytest.raises(RuntimeError, match="Remove Testing Farm"):
        REVISION.downgrade()
    assert operations.statements == [
        ("drop_index", "ix_task_analysis_deadline_at", "task_analysis"),
        ("drop_column", "task_analysis", "deadline_at"),
    ]


def test_tft_migration_downgrade_removes_both_labels(monkeypatch):
    """A safe rollback restores both enum definitions."""
    operations = FakeOperations(in_use=False)
    monkeypatch.setattr(REVISION, "op", operations)
    REVISION.downgrade()
    assert operations.statements == [
        ("drop_index", "ix_task_analysis_deadline_at", "task_analysis"),
        ("drop_column", "task_analysis", "deadline_at"),
        "ALTER TYPE tasktype RENAME TO tasktype_with_suite",
        "CREATE TYPE tasktype AS ENUM ('generic', 'koji', 'gitlab')",
        "ALTER TABLE task_analysis ALTER COLUMN task_type TYPE tasktype "
        "USING task_type::text::tasktype",
        "DROP TYPE tasktype_with_suite",
        "ALTER TYPE endpointtype RENAME TO endpointtype_with_tft",
        "CREATE TYPE endpointtype AS ENUM "
        "('analyze', 'analyze_staged', 'analyze_stream', "
        "'analyze_gitlab_job', 'analyze_koji_task')",
        "ALTER TABLE analyze_request_metrics ALTER COLUMN endpoint "
        "TYPE endpointtype USING endpoint::text::endpointtype",
        "DROP TYPE endpointtype_with_tft",
    ]


def test_fresh_postgres_schema_migrates_to_suite_deadline():
    """Apply the unreleased revision as part of a fresh PostgreSQL upgrade."""
    schema = f"tft_fresh_{uuid4().hex}"
    conninfo = "postgresql://user:password@localhost:5432/test_db"
    with psycopg.connect(conninfo, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            env = os.environ.copy()
            env.update({
                "POSTGRESQL_USER": "user", "POSTGRESQL_PASSWORD": "password",
                "POSTGRESQL_DATABASE": "test_db", "POSTGRESQL_HOST": "localhost",
                "POSTGRESQL_PORT": "5432",
                "PGOPTIONS": f"-c search_path={schema},public",
            })
            result = subprocess.run(
                [sys.executable, "-m", "alembic", "upgrade", "head"],
                cwd=REVISION_PATH.parent.parent.parent, env=env,
                capture_output=True, text=True, timeout=120, check=False,
            )
            assert result.returncode == 0, result.stderr
            column = connection.execute(
                "SELECT 1 FROM information_schema.columns WHERE table_schema = %s "
                "AND table_name = 'task_analysis' AND column_name = 'deadline_at'",
                (schema,),
            ).fetchone()
            assert column == (1,)
        finally:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(
                sql.Identifier(schema)
            ))


def test_database_models_do_not_import_server_runtime():
    """Migration metadata imports must not require the server dependency graph."""
    result = subprocess.run(
        [sys.executable, "-c", '''
import importlib.abc
import sys
class NoServerRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "logdetective.config":
            raise ImportError("Migration metadata must not load server configuration")
sys.meta_path.insert(0, NoServerRuntime())
import logdetective.database.models
'''],
        cwd=REVISION_PATH.parent.parent.parent,
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr
