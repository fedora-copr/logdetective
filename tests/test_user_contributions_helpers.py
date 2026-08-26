"""
Test fixtures and helper functions for test_user_contribution_update.
Unit tests for ContributionBuild Pydantic validation.
"""

import contextlib
import io
import json
import tarfile
from datetime import date
from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError

from logdetective.constants import EMBEDDING_VECTOR_SIZE
from logdetective.models import ContributionBuild


# test fixtures for test_user_contributions_update.py


ZERO_EMBEDDING = np.zeros(EMBEDDING_VECTOR_SIZE, dtype=np.float32)

CONTRIBUTION_VALID = {
    "fail_reason": "Missing dependency libfoo.",
    "how_to_fix": "Add BuildRequires: libfoo-devel.",
    "spec_file": None,
    "logs": {
        "builder-live.log": {
            "name": "builder-live.log", "content": "log",
            "snippets": [
                {
                    "start_index": 0,
                    "end_index": 50,
                    "text": "error: libfoo not found",
                    "user_comment": "Build cannot find libfoo."
                },
                {
                    "start_index": 100,
                    "end_index": 150,
                    "text": "FAILED test_bar_function: assert 0 == 1",
                    "user_comment": "One test from the test suite failed."
                },
            ],
        },
        "build.log": {
            "name": "build.log", "content": "log",
            "snippets": [
                {
                    "start_index": 0,
                    "end_index": 30,
                    "text": "make: *** Error 1",
                    "user_comment": "Build aborted."
                },
            ],
        },
    },
}

CONTRIBUTION_SECOND = {
    "fail_reason": "Syntax error in main.c.",
    "how_to_fix": "Fix line 42.",
    "spec_file": None,
    "logs": {
        "build.log": {
            "name": "build.log", "content": "log",
            "snippets": [
                {
                    "start_index": 0,
                    "end_index": 30,
                    "text": "main.c:42: error: expected ';'",
                    "user_comment": "Missing semicolon."
                },
            ],
        },
    },
}

CONTRIBUTION_MALFORMED = {"not_a": "real contribution"}

MULTI_SNIPPET_CONTRIBUTION = {
    "fail_reason": "Missing dependency libfoo.",
    "how_to_fix": "Add BuildRequires: libfoo-devel to the spec file.",
    "spec_file": None,
    "logs": {
        "builder-live.log": {
            "name": "builder-live.log",
            "content": "log content 1 ...",
            "snippets": [
                {
                    "start_index": 0,
                    "end_index": 100,
                    "text": "error: libfoo not found",
                    "user_comment": "Build cannot find libfoo.",
                },
                {
                    "start_index": 200,
                    "end_index": 300,
                    "text": "FAILED test_bar_function: assert 0 == 1",
                    "user_comment": "One test from the test suite failed.",
                },
            ],
        },
        "build.log": {
            "name": "build.log",
            "content": "log content 2 ...",
            "snippets": [
                {
                    "start_index": 0, "end_index": 50,
                    "text": "make: *** [Makefile:42] Error 1",
                    "user_comment": "Build aborted due to missing dependency.",
                },
            ],
        },
    },
}


# helper functions for integration tests in test_user_contributions_update.py


def _build_tar_gz(files: dict[str, dict]) -> bytes:
    """Build a tar.gz archive in memory from {path: json_data}."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path, data in files.items():
            content = json.dumps(data).encode()
            info = tarfile.TarInfo(name=f"results/results/{path}")
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    buf.seek(0)
    return buf.read()


def _mock_download_factory(archives: list[tuple[bytes, str]]):
    """Returns a mock for _download_to_spool that pops archives in order."""
    remaining = list(archives)

    async def mock_download_to_spool(spool, url, params):
        data, name = remaining.pop(0)
        spool.write(data)
        spool.seek(0)
        return name

    return mock_download_to_spool


def _frozen_date_patches(frozen_date):
    """Patch date.today() in both modules so tests are time-agnostic."""
    kwargs = {"today": lambda: frozen_date, "fromisoformat": date.fromisoformat}
    return [
        patch("logdetective.user_contributions_update.date", **kwargs),
        patch("logdetective.user_contributions_helpers.date", **kwargs),
    ]


def mock_network_and_embeddings(mock_dl, frozen_date):
    """
    Patch everything needed to run run_update() without network or real models.

    Patches: HTTP download, embedding model, date.today() in both modules.
    Returns an ExitStack so the caller can activate all patches with a single
    `with` statement instead of nesting one `with` per patch.
    """
    stack = contextlib.ExitStack()
    stack.enter_context(patch(
        "logdetective.user_contributions_helpers._download_to_spool",
        side_effect=mock_dl,
    ))
    stack.enter_context(patch(
        "logdetective.user_contributions_update.TextEmbedding",
        return_value=type("M", (), {
            "embed": lambda self, texts, **kw: iter(
                [ZERO_EMBEDDING] * len(texts)
            ),
        })(),
    ))
    for p in _frozen_date_patches(frozen_date):
        stack.enter_context(p)
    return stack


def mock_network_only(mock_dl, frozen_date):
    """
    Like mock_network_and_embeddings but uses the real embedding model.

    Patches: HTTP download, date.today() in both modules.
    Returns an ExitStack so the caller can activate all patches with a single
    `with` statement instead of nesting one `with` per patch.
    """
    stack = contextlib.ExitStack()
    stack.enter_context(patch(
        "logdetective.user_contributions_helpers._download_to_spool",
        side_effect=mock_dl,
    ))
    for p in _frozen_date_patches(frozen_date):
        stack.enter_context(p)
    return stack


# ContributionBuild Pydantic validation unit tests


def test_contribution_build_multi_snippet():
    """Valid contribution with multiple snippets across multiple log files."""
    result = ContributionBuild(**MULTI_SNIPPET_CONTRIBUTION)
    all_snippets = [
        (log_name, snippet)
        for log_name, entry in result.logs.items()
        for snippet in entry.snippets
    ]
    assert len(all_snippets) == 3, "3 snippets across 2 log files"
    assert {log_name for log_name, _ in all_snippets} == {"builder-live.log", "build.log"}
    assert all(s.text and s.user_comment for _, s in all_snippets)


def test_contribution_build_rejects_malformed():
    """Malformed contributions raise ValidationError or TypeError."""
    with pytest.raises((ValidationError, TypeError)):
        ContributionBuild(**{"not": "a contribution"})

    with pytest.raises(ValidationError):
        ContributionBuild(**{**MULTI_SNIPPET_CONTRIBUTION, "fail_reason": {"text": "foo"}})

    with pytest.raises(ValidationError):
        ContributionBuild(**{**MULTI_SNIPPET_CONTRIBUTION, "logs": [1, 2, 3]})

    with pytest.raises(ValidationError):
        ContributionBuild(**{**MULTI_SNIPPET_CONTRIBUTION, "fail_reason": ""})

    with pytest.raises(ValidationError):
        ContributionBuild(**{**MULTI_SNIPPET_CONTRIBUTION, "how_to_fix": ""})


def test_contribution_build_rejects_incomplete():
    """Contributions with missing/empty snippets raise ValidationError."""
    with pytest.raises(ValidationError):
        ContributionBuild(**{
            **MULTI_SNIPPET_CONTRIBUTION,
            "logs": {"build.log": {"name": "build.log", "content": "log", "snippets": []}},
        })

    with pytest.raises(ValidationError):
        ContributionBuild(**{
            **MULTI_SNIPPET_CONTRIBUTION,
            "logs": {"build.log": {
                "name": "build.log", "content": "log",
                "snippets": [{"text": "some error"}],
            }},
        })

    with pytest.raises(ValidationError):
        ContributionBuild(**{
            **MULTI_SNIPPET_CONTRIBUTION,
            "logs": {"build.log": {
                "name": "build.log", "content": "log",
                "snippets": [{"user_comment": "a comment"}],
            }},
        })

    with pytest.raises(ValidationError):
        ContributionBuild(**{
            **MULTI_SNIPPET_CONTRIBUTION,
            "logs": {"build.log": {
                "name": "build.log", "content": "log",
                "snippets": [{"text": "", "user_comment": ""}],
            }},
        })
