"""Integration tests for the DB update script with mock archives and real DB."""

from datetime import date

import pytest
from fastembed import TextEmbedding
from sqlalchemy import select

from logdetective.constants import EMBEDDING_MODEL, EMBEDDING_VECTOR_SIZE
from logdetective.server.database.base import transaction
from logdetective.server.user_contributions_update import run_update
from logdetective.server.database.models.annotated_builds import (
    AnnotatedBuilds,
    AnnotatedSnippets,
    AnnotationUpdates,
)
from tests.server.test_helpers import DatabaseFactory
from tests.server.test_user_contributions_helpers import (
    ZERO_EMBEDDING,
    CONTRIBUTION_VALID,
    CONTRIBUTION_SECOND,
    CONTRIBUTION_MALFORMED,
    _build_tar_gz,
    _mock_download_factory,
    mock_network_and_embeddings,
    mock_network_only,
)


@pytest.mark.asyncio
async def test_bootstrap_with_valid_and_malformed():
    """
    Bootstrap: full archive has 1 valid (3 snippets) + 1 malformed contribution.
    Delta has 1 new contribution. Malformed is skipped, rest inserted.
    """
    full = _build_tar_gz({
        "2025-01-01/copr/111/aaa.json": CONTRIBUTION_VALID,
        "2025-01-01/copr/222/bad.json": CONTRIBUTION_MALFORMED,
    })
    delta = _build_tar_gz({
        "2025-07-10/copr/333/bbb.json": CONTRIBUTION_SECOND,
    })
    dl = _mock_download_factory([
        (full, "results-2025-07-01.tar.gz"),
        (delta, "results-since-2025-07-01.tar.gz"),
    ])

    db = DatabaseFactory()
    async with db.make_new_db():
        with mock_network_and_embeddings(dl, date(2025, 7, 15)):
            await run_update("https://example.com/download")

        assert await AnnotatedBuilds.get_count() == 2, \
            "2 valid builds (malformed skipped)"
        assert await AnnotatedSnippets.get_count() == 4, \
            "3 snippets from first + 1 from second"
        assert await AnnotationUpdates.get_latest_date() == date(2025, 7, 15)

        # Verify build content
        async with transaction(commit=False) as session:
            builds = (await session.execute(
                select(AnnotatedBuilds).order_by(AnnotatedBuilds.id)
            )).scalars().all()

            assert builds[0].problem == "Missing dependency libfoo."
            assert builds[0].solution == "Add BuildRequires: libfoo-devel."
            assert builds[0].source_path == "results/results/2025-01-01/copr/111/aaa.json"
            assert builds[1].problem == "Syntax error in main.c.", "build 2 problem"
            assert builds[1].solution == "Fix line 42."
            assert builds[1].source_path == "results/results/2025-07-10/copr/333/bbb.json"

            # Verify snippet content and embeddings
            snippets = (await session.execute(
                select(AnnotatedSnippets).order_by(AnnotatedSnippets.id)
            )).scalars().all()

            assert snippets[0].text == "error: libfoo not found"
            assert snippets[0].annotation == "Build cannot find libfoo."
            assert snippets[0].source_artifact_name == "builder-live.log"

            assert snippets[1].text == "FAILED test_bar_function: assert 0 == 1"
            assert snippets[1].annotation == "One test from the test suite failed."
            assert snippets[1].source_artifact_name == "builder-live.log"

            assert snippets[0].source_build_id == builds[0].id, \
                "snippet 1 linked to build 1"
            assert snippets[2].source_artifact_name == "build.log", \
                "snippet 3 from different log file"
            assert snippets[2].source_build_id == builds[0].id, \
                "snippet 3 still linked to build 1"
            assert snippets[3].source_build_id == builds[1].id, \
                "snippet 4 linked to build 2"

            for s in snippets:
                embedding = list(s.text_embedding)
                msg = (
                    f"snippet {s.id} embedding has {len(embedding)} dims, "
                    f"expected {EMBEDDING_VECTOR_SIZE}"
                )
                assert len(embedding) == EMBEDDING_VECTOR_SIZE, msg

        # Verify vector similarity lookup returns snippets with linked builds
        results = await AnnotatedSnippets.get_by_snippet_embedding(
            ZERO_EMBEDDING.tolist(), top_k=4,
        )
        assert len(results) == 4, "lookup returns all 4 snippets"
        for r in results:
            assert r.text, "lookup result has text"
            assert r.annotation, "lookup result has annotation"
            assert r.source_build is not None, "lookup eagerly loads source_build"
            assert r.source_build.problem, "linked build has problem"
            assert r.source_build.solution, "linked build has solution"


@pytest.mark.asyncio
async def test_incremental_skips_duplicates():
    """Test that in update with 1 duplicate + 1 new, only the new contribution is inserted."""
    full = _build_tar_gz({
        "2025-01-01/copr/111/aaa.json": CONTRIBUTION_VALID,
    })
    empty_delta = _build_tar_gz({})
    incremental = _build_tar_gz({
        "2025-01-01/copr/111/aaa.json": CONTRIBUTION_VALID,
        "2025-07-20/copr/333/ccc.json": CONTRIBUTION_SECOND,
    })

    db = DatabaseFactory()
    async with db.make_new_db():
        dl = _mock_download_factory([
            (full, "results-2025-07-15.tar.gz"),
            (empty_delta, "results-since-2025-07-15.tar.gz"),
        ])
        with mock_network_and_embeddings(dl, date(2025, 7, 20)):
            await run_update("https://example.com/download")

        assert await AnnotatedBuilds.get_count() == 1, "1 build after bootstrap"

        dl = _mock_download_factory([
            (incremental, "results-since-2025-07-20.tar.gz"),
        ])
        with mock_network_and_embeddings(dl, date(2025, 7, 25)):
            await run_update("https://example.com/download")

        assert await AnnotatedBuilds.get_count() == 2, "duplicate skipped, new one added"
        total = await AnnotationUpdates.get_total_files_processed()
        assert total > 2, "total includes the skipped duplicate"


@pytest.mark.asyncio
async def test_noop_updates():
    """Duplicate archive and empty archive - DB state unchanged."""
    full = _build_tar_gz({
        "2025-01-01/copr/111/aaa.json": CONTRIBUTION_VALID,
    })
    empty_delta = _build_tar_gz({})
    all_dupes = _build_tar_gz({
        "2025-01-01/copr/111/aaa.json": CONTRIBUTION_VALID,
    })

    db = DatabaseFactory()
    async with db.make_new_db():
        # Bootstrap
        dl = _mock_download_factory([
            (full, "results-2025-07-15.tar.gz"),
            (empty_delta, "results-since-2025-07-15.tar.gz"),
        ])
        with mock_network_and_embeddings(dl, date(2025, 7, 20)):
            await run_update("https://example.com/download")

        builds_after_bootstrap = await AnnotatedBuilds.get_count()
        assert builds_after_bootstrap == 1

        # Run with all-duplicate archive (jsons_processed > 0 but nothing new)
        dl = _mock_download_factory([
            (all_dupes, "results-since-2025-07-20.tar.gz"),
        ])
        with mock_network_and_embeddings(dl, date(2025, 7, 25)):
            await run_update("https://example.com/download")

        assert await AnnotatedBuilds.get_count() == 1, "no new builds from duplicates"

        # Run with empty archive (jsons_processed = 0)
        dl = _mock_download_factory([
            (empty_delta, "results-since-2025-07-25.tar.gz"),
        ])
        with mock_network_and_embeddings(dl, date(2025, 7, 30)):
            await run_update("https://example.com/download")

        assert await AnnotatedBuilds.get_count() == 1, "no new builds from empty archive"


@pytest.mark.asyncio
async def test_dry_run_and_reset():
    """Dry run makes no DB changes. Reset wipes and repopulates."""
    full_v1 = _build_tar_gz({
        "2025-01-01/copr/111/aaa.json": CONTRIBUTION_VALID,
    })
    full_v2 = _build_tar_gz({
        "2025-07-20/copr/222/bbb.json": CONTRIBUTION_SECOND,
    })
    empty_delta = _build_tar_gz({})

    db = DatabaseFactory()
    async with db.make_new_db():
        # Dry run - nothing inserted
        dl = _mock_download_factory([
            (full_v1, "results-2025-07-01.tar.gz"),
            (empty_delta, "results-since-2025-07-01.tar.gz"),
        ])
        with mock_network_and_embeddings(dl, date(2025, 7, 15)):
            await run_update("https://example.com/download", dry_run=True)

        assert await AnnotatedBuilds.get_count() == 0, "dry run: no builds"
        assert await AnnotationUpdates.get_latest_date() is None, "dry run: no records"

        # Real bootstrap
        dl = _mock_download_factory([
            (full_v1, "results-2025-07-01.tar.gz"),
            (empty_delta, "results-since-2025-07-01.tar.gz"),
        ])
        with mock_network_and_embeddings(dl, date(2025, 7, 15)):
            await run_update("https://example.com/download")

        assert await AnnotatedBuilds.get_count() == 1

        # Reset with different data
        dl = _mock_download_factory([
            (full_v2, "results-2025-07-20.tar.gz"),
            (empty_delta, "results-since-2025-07-20.tar.gz"),
        ])
        with mock_network_and_embeddings(dl, date(2025, 7, 25)):
            await run_update("https://example.com/download", reset=True)

        assert await AnnotatedBuilds.get_count() == 1, "old wiped, new inserted"


@pytest.mark.asyncio
async def test_semantic_lookup_with_real_embeddings():
    """
    End-to-end: insert contributions with real embeddings, then query with
    a semantically similar string and verify the right snippet is returned
    with its linked build.
    """

    full = _build_tar_gz({
        "2025-01-01/copr/111/aaa.json": CONTRIBUTION_VALID,
        "2025-07-10/copr/333/bbb.json": CONTRIBUTION_SECOND,
    })
    empty_delta = _build_tar_gz({})
    dl = _mock_download_factory([
        (full, "results-2025-07-01.tar.gz"),
        (empty_delta, "results-since-2025-07-01.tar.gz"),
    ])

    db = DatabaseFactory()
    async with db.make_new_db():
        with mock_network_only(dl, date(2025, 7, 15)):
            await run_update("https://example.com/download")

        # Query with something semantically close to "error: libfoo not found"
        model = TextEmbedding(EMBEDDING_MODEL)
        query_embedding = list(model.embed(["missing library foo dependency"]))[0]

        results = await AnnotatedSnippets.get_by_snippet_embedding(
            query_embedding.tolist(), top_k=1,
        )
        assert len(results) == 1, "lookup returns a result"

        top = results[0]
        assert "libfoo" in top.text, \
            f"closest match should be the libfoo snippet, got: {top.text}"
        assert top.source_build is not None, "eagerly loaded source_build"
        assert "libfoo" in top.source_build.problem, \
            f"linked build should mention libfoo, got: {top.source_build.problem}"
