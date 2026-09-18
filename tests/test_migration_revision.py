"""Regression tests for task enum migration behavior."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from sqlalchemy import Enum
from sqlalchemy.dialects import postgresql


REVISION_PATH = Path(__file__).parent.parent.joinpath(
    "alembic", "versions", "a090e4500d54_procrastinate_analysis_tasks.py"
)
REVISION_SPEC = spec_from_file_location("task_analysis_revision", REVISION_PATH)
if REVISION_SPEC is None or REVISION_SPEC.loader is None:
    raise RuntimeError(f"Unable to load migration revision {REVISION_PATH}")
REVISION = module_from_spec(REVISION_SPEC)
REVISION_SPEC.loader.exec_module(REVISION)


class FakeBind:  # pylint: disable=too-few-public-methods
    """Report that the replaced task table contains no records."""

    def __init__(self, scalar_results: list[bool] | None = None) -> None:
        self.scalar_results = iter(scalar_results or [False])

    def scalar(self, _statement) -> bool:
        return next(self.scalar_results)


class FakeOperations:  # pylint: disable=too-many-instance-attributes
    """Capture migration DDL and the replacement table definition."""

    def __init__(self, scalar_results: list[bool] | None = None) -> None:
        self.bind = FakeBind(scalar_results)
        self.statements: list[str] = []
        self.columns = []

    def get_bind(self) -> FakeBind:
        return self.bind

    def execute(self, statement: str) -> None:
        self.statements.append(statement)

    @staticmethod
    def drop_table(_name: str) -> None:
        return None

    def create_table(self, _name: str, *elements) -> None:
        self.columns = [element for element in elements if hasattr(element, "type")]

    @staticmethod
    def create_index(*_args, **_kwargs) -> None:
        return None


def capture_enum_creates(monkeypatch) -> list[tuple[str, list[str]]]:
    """Capture native enum creation without requiring a live PostgreSQL bind."""
    created_types: list[tuple[str, list[str]]] = []

    def record_create(enum_type, _bind, checkfirst=False) -> None:
        assert checkfirst is False
        created_types.append((enum_type.name, enum_type.enums))

    monkeypatch.setattr(postgresql.ENUM, "create", record_create)
    return created_types


def test_upgrade_uses_lowercase_enum_labels(monkeypatch):
    """The replacement schema stores every application enum as lowercase."""
    operations = FakeOperations()
    monkeypatch.setattr(REVISION, "op", operations)
    created_types = capture_enum_creates(monkeypatch)

    REVISION.upgrade()

    assert operations.statements == [
        "DROP TYPE IF EXISTS tasktype;",
        "DROP TYPE IF EXISTS analysisstate;",
        "ALTER TYPE endpointtype RENAME TO endpointtype_uppercase;",
        "ALTER TABLE analyze_request_metrics ALTER COLUMN endpoint "
        "TYPE endpointtype USING lower(endpoint::text)::endpointtype;",
        "DROP TYPE endpointtype_uppercase;",
    ]
    assert created_types == [
        (
            "analysisstate",
            [
                "scheduled",
                "in_progress",
                "cancelling",
                "cancelled",
                "done",
                "error",
            ],
        ),
        ("tasktype", ["generic", "koji", "gitlab"]),
        (
            "endpointtype",
            [
                "analyze",
                "analyze_staged",
                "analyze_stream",
                "analyze_gitlab_job",
                "analyze_koji_task",
            ],
        ),
    ]
    enums = {
        column.name: column.type.enums
        for column in operations.columns
        if isinstance(column.type, Enum)
    }
    assert enums == {
        "task_type": ["generic", "koji", "gitlab"],
        "state": [
            "scheduled",
            "in_progress",
            "cancelling",
            "cancelled",
            "done",
            "error",
        ],
    }


def test_downgrade_restores_uppercase_enum_labels(monkeypatch):
    """Downgrade recreates the exact enum labels used by the former schema."""
    operations = FakeOperations([False, False])
    monkeypatch.setattr(REVISION, "op", operations)
    created_types = capture_enum_creates(monkeypatch)

    REVISION.downgrade()

    assert operations.statements == [
        "DROP TYPE tasktype;",
        "DROP TYPE analysisstate;",
        "ALTER TYPE endpointtype RENAME TO endpointtype_lowercase;",
        "ALTER TABLE analyze_request_metrics ALTER COLUMN endpoint "
        "TYPE endpointtype USING upper(endpoint::text)::endpointtype;",
        "DROP TYPE endpointtype_lowercase;",
    ]
    assert created_types == [
        ("analysisstate", ["SCHEDULED", "DONE", "IN_PROGRESS", "ERROR"]),
        ("tasktype", ["GENERIC", "KOJI"]),
        (
            "endpointtype",
            ["ANALYZE", "ANALYZE_STAGED", "ANALYZE_STREAM", "ANALYZE_GITLAB_JOB"],
        ),
    ]
    enums = {
        column.name: column.type.enums
        for column in operations.columns
        if isinstance(column.type, Enum)
    }
    assert enums == {
        "task_type": ["GENERIC", "KOJI"],
        "state": ["SCHEDULED", "DONE", "IN_PROGRESS", "ERROR"],
    }


def test_downgrade_refuses_to_discard_koji_metrics(monkeypatch):
    """The former endpoint enum cannot represent Koji task metrics."""
    operations = FakeOperations([False, True])
    monkeypatch.setattr(REVISION, "op", operations)

    with pytest.raises(RuntimeError, match="Koji metrics exist"):
        REVISION.downgrade()

    assert not operations.statements
