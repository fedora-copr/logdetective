"""Tests for AnnotatedBuilds, AnnotatedSnippets, and AnnotationUpdates models."""

from datetime import date

import numpy as np
import pytest

from logdetective.constants import EMBEDDING_VECTOR_SIZE
from logdetective.database.models.annotated_builds import (
    AnnotatedBuilds,
    AnnotatedSnippets,
    AnnotationUpdates,
)
from tests.server.test_helpers import DatabaseFactory


ZERO_EMBEDDING = np.zeros(EMBEDDING_VECTOR_SIZE, dtype=np.float32).tolist()


@pytest.mark.asyncio
async def test_annotated_builds_create_and_count():
    db = DatabaseFactory()
    async with db.make_new_db():
        count = await AnnotatedBuilds.get_count()
        assert count == 0, "empty table"

        build_id = await AnnotatedBuilds.create(
            problem="test problem",
            solution="test solution",
            source_path="2025-01-01/copr/111/aaa.json",
        )
        assert build_id is not None, "create returns an id"

        count = await AnnotatedBuilds.get_count()
        assert count == 1, "one build inserted"


@pytest.mark.asyncio
async def test_annotated_builds_source_path_unique():
    db = DatabaseFactory()
    async with db.make_new_db():
        await AnnotatedBuilds.create(
            problem="p1", solution="s1",
            source_path="2025-01-01/copr/111/aaa.json",
        )
        with pytest.raises(Exception):
            await AnnotatedBuilds.create(
                problem="p2", solution="s2",
                source_path="2025-01-01/copr/111/aaa.json",
            )


@pytest.mark.asyncio
async def test_annotated_snippets_create_and_count():
    db = DatabaseFactory()
    async with db.make_new_db():
        build_id = await AnnotatedBuilds.create(
            problem="test problem",
            solution="test solution",
            source_path="./mock_path.json",
        )

        snippet_id = await AnnotatedSnippets.create(
            text="error: libfoo not found",
            annotation="The build cannot find libfoo.",
            text_embedding=ZERO_EMBEDDING,
            source_artifact_name="builder-live.log",
            source_build_id=build_id,
        )
        assert snippet_id is not None, "create returns an id"

        count = await AnnotatedSnippets.get_count()
        assert count == 1, "one snippet inserted"


@pytest.mark.asyncio
async def test_annotation_updates_lifecycle():
    db = DatabaseFactory()
    async with db.make_new_db():
        latest = await AnnotationUpdates.get_latest_date()
        assert latest is None, "empty table returns None"

        total = await AnnotationUpdates.get_total_files_processed()
        assert total == 0, "empty table returns 0"

        await AnnotationUpdates.add_update_record(
            file_count=10, archive_date=date(2025, 7, 1),
        )
        await AnnotationUpdates.add_update_record(
            file_count=5, archive_date=date(2025, 7, 8),
        )

        latest = await AnnotationUpdates.get_latest_date()
        assert latest == date(2025, 7, 8), "returns most recent date"

        total = await AnnotationUpdates.get_total_files_processed()
        assert total == 15, "sums all file counts"
