"""
Update the vector database with user contributions from logdetective.com.

Packagers contribute build failure analyses on logdetective.com - each contribution
contains a fail_reason, how_to_fix, and annotated log snippets. This script downloads
those contributions and populates the pgvector database (AnnotatedBuilds and
AnnotatedSnippets tables) so the Log Detective Agent can use them as a RAG search
tool to improve analysis precision.

How it works:
  - On first run (or --reset): downloads the full archive via /download, inserts it,
    then catches up with recent contributions via /download?since=<archive_date>.
    A single AnnotationUpdates record is created with the combined file count.
  - On subsequent runs: fetches only new contributions since the last update
    via /download?since=<last_update_date>.
  - Duplicate contributions are skipped at the DB level via the unique constraint
    on source_path - path to JSON contribution file within the archive
    (e.g. 2025-01-01/copr/123456/abcdef.json).
  - DB writes for each insertion step happen in an atomic transaction.
    During bootstrap, there are two separate transactions (full + delta).
  - Each snippet text retrieved via RAG is then related to a snippet comment,
    and the related build failure explanation and proposed fix.
  - --dry-run does everything except the database insertion/deletion.

Usage:
  python -m logdetective.server.user_contributions_update
  python -m logdetective.server.user_contributions_update --dry-run
  python -m logdetective.server.user_contributions_update --reset

    Expected to run as a monthly cronjob (in an OC cluster/compose).

When running in compose, make sure postgres container is running and then:
`podman-compose --profile sync run --rm user-contributions-update`
Logs from the script should be saved to the path specified by LOG_DIR/LOG_FILE:
    - /var/lib/logdetective/user_contributions_update.log
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import date
from typing import Iterator, Optional, Tuple

import backoff
import numpy as np
from fastembed import TextEmbedding
from pydantic import ValidationError
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import OperationalError

from logdetective.constants import EMBEDDING_MODEL
from logdetective.server.database.base import (
    transaction,
    DB_MAX_RETRIES,
)
from logdetective.server.database.models.annotated_builds import (
    AnnotatedBuilds,
    AnnotatedSnippets,
    AnnotationUpdates,
)
from logdetective.server.models import ContributionBuild
from logdetective.server.user_contributions_helpers import (
    CurrentUpdateStats,
    ValidatedBuild,
    sanitize_for_pg,
    extract_archive_date,
    log_update_report,
    reset_tables,
    fetch_and_parse,
)

logger = logging.getLogger(__name__)


DOWNLOAD_URL = "https://logdetective.com/download"
MAX_ARCHIVE_AGE_DAYS = 90
LOG_DIR = os.getenv("LOG_DIR", "/var/log/logdetective")
LOG_FILE = os.path.join(LOG_DIR, "user_contributions_update.log")


# pylint: disable=too-many-locals
@backoff.on_exception(backoff.expo, OperationalError, max_tries=DB_MAX_RETRIES)
async def insert_contributions(
    builds: list[ValidatedBuild],
    embeddings: list[np.ndarray],
    stats: CurrentUpdateStats,
) -> None:
    """
    Bulk-insert builds and snippets in a single atomic transaction.
    Duplicates (by source_path) are skipped via ON CONFLICT DO NOTHING.
    Embeddings must match the flattened snippet order across all builds.
    """
    if not builds:
        return

    build_values: list[dict] = []
    builds_by_path: dict[str, tuple[ValidatedBuild, int]] = {}
    offset = 0
    for b in builds:
        build_values.append({
            "problem": sanitize_for_pg("fail_reason", b.fail_reason),
            "solution": sanitize_for_pg("how_to_fix", b.how_to_fix),
            "source_path": b.source_path,
        })
        builds_by_path[b.source_path] = (b, offset)
        offset += len(b.snippets)

    async with transaction(commit=True) as session:
        build_stmt = (
            pg_insert(AnnotatedBuilds)
            .values(build_values)
            .on_conflict_do_nothing(index_elements=["source_path"])
            .returning(AnnotatedBuilds.id, AnnotatedBuilds.source_path)
        )
        inserted_builds = (await session.execute(build_stmt)).all()

        snippet_values = []
        for build_id, source_path in inserted_builds:
            build, emb_offset = builds_by_path[source_path]
            for i, s in enumerate(build.snippets):
                snippet_values.append({
                    "text": sanitize_for_pg("snippet.text", s.text),
                    "annotation": sanitize_for_pg("snippet.user_comment", s.user_comment),
                    "text_embedding": embeddings[emb_offset + i].tolist(),
                    "source_artifact_name": sanitize_for_pg("snippet.log_name", s.log_name),
                    "source_build_id": build_id,
                })

        if snippet_values:
            await session.execute(pg_insert(AnnotatedSnippets).values(snippet_values))

    stats.builds_added += len(inserted_builds)
    stats.snippets_added += len(snippet_values)
    logger.info(
        "Inserted %d builds (skipped %d) with %d snippets -- Total: %d builds, %d snippets",
        len(inserted_builds),
        len(builds) - len(inserted_builds),
        len(snippet_values),
        stats.builds_added,
        stats.snippets_added,
    )


async def validate_and_insert_contributions(
    contributions: Iterator[Tuple[str, dict]],
    embedding_model: Optional[TextEmbedding],
    stats: CurrentUpdateStats,
    dry_run: bool = False,
) -> None:
    """Validate, embed, and insert contributions in batches."""
    total_counter = 0
    valid_counter = 0

    current_batch: list[ValidatedBuild] = []
    current_snippet_count = 0

    for relative_path, contribution_data in contributions:
        total_counter += 1
        try:
            build = ValidatedBuild(
                relative_path,
                ContributionBuild(**contribution_data)
            )
            valid_counter += 1
        except ValidationError as exc:
            err = exc.errors()[0]
            logger.warning(
                "Skipping %s: %s - %s", relative_path,
                ".".join(str(loc) for loc in err["loc"]),
                err["msg"].removeprefix("Value error, ")
            )
            continue
        except TypeError as exc:
            logger.warning(
                "Skipping %s: %s", relative_path, exc
            )
            continue

        num_snippets = len(build.snippets)

        if current_batch and (current_snippet_count + num_snippets > 64):
            await _process_batch(current_batch, embedding_model, stats, dry_run)
            current_batch = []
            current_snippet_count = 0

        current_batch.append(build)
        current_snippet_count += num_snippets

    if current_batch:
        await _process_batch(current_batch, embedding_model, stats, dry_run)

    logger.info(
        "Validation done: %d (valid) / %d (total) contributions",
        valid_counter,
        total_counter,
    )


async def _process_batch(
    builds: list[ValidatedBuild],
    embedding_model: Optional[TextEmbedding],
    stats: CurrentUpdateStats,
    dry_run: bool,
) -> None:
    if not builds:
        return

    builds_in_batch = len(builds)
    snips_in_batch = sum(len(b.snippets) for b in builds)
    max_snip_length = max(len(s.text) for b in builds for s in b.snippets)
    logger.info(
        "Batch%s: %d builds, %d snippets (longest snippet = %d chars)",
        " (dry run)" if dry_run else "",
        builds_in_batch,
        snips_in_batch,
        max_snip_length,
    )

    if dry_run:
        stats.builds_added += builds_in_batch
        stats.snippets_added += snips_in_batch
        return

    all_snippet_texts = [s.text for b in builds for s in b.snippets]

    if not all_snippet_texts:
        embeddings = []
    else:
        embedding_generator = embedding_model.embed(
            all_snippet_texts, batch_size=16
        )
        embeddings = (
            embedding_generator
            if isinstance(embedding_generator, list)
            else list(embedding_generator)
        )

    await insert_contributions(builds, embeddings, stats)


async def run_update(url: str, dry_run: bool = False, reset: bool = False) -> None:
    """Main update orchestrator."""

    logger.info("Mode: dry_run=%s, reset=%s", dry_run, reset)

    try:
        date_since = await AnnotationUpdates.get_latest_date()
    except Exception as exc:
        logger.error("Failed to connect to database: %s. Check POSTGRESQL_* env vars.", exc)
        raise

    embedding_model = None
    if not dry_run:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        embedding_model = TextEmbedding(EMBEDDING_MODEL)

    if reset:
        date_since = None
        if dry_run:
            logger.info("Dry run - would wipe all annotation tables.")
        else:
            logger.warning("Reset requested - wiping existing annotation data.")
            await reset_tables()

    stats = CurrentUpdateStats()
    if not date_since:
        logger.info("Downloading full archive.")

        async with fetch_and_parse(url, stats) as (contributions, filename):
            if not filename:
                raise RuntimeError(
                    "Server did not provide an archive filename via Content-Disposition. "
                    "Cannot determine archive date for catch-up download."
                )

            date_since = extract_archive_date(filename)
            if (age_days := (date.today() - date_since).days) > MAX_ARCHIVE_AGE_DAYS:
                raise RuntimeError(
                    f"Full archive date {date_since} is {age_days} days old "
                    f"(>{MAX_ARCHIVE_AGE_DAYS}). The cached archive on logdetective.com "
                    f"needs to be regenerated before the catch-up download will work."
                )

            await validate_and_insert_contributions(
                contributions, embedding_model, stats, dry_run,
            )

    logger.info("Incremental sync since %s.", date_since)
    async with fetch_and_parse(url, stats, since=date_since) as (contributions_since, _):
        await validate_and_insert_contributions(
            contributions_since, embedding_model, stats, dry_run,
        )

    if not dry_run and stats.jsons_processed > 0:
        await AnnotationUpdates.add_update_record(
            file_count=stats.jsons_processed, archive_date=date.today(),
        )

    await log_update_report(stats, dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sync user contributions from logdetective.com into the vector database",
    )
    parser.add_argument(
        "--url",
        default=DOWNLOAD_URL,
        help=f"URL to download contributions archive (default: {DOWNLOAD_URL})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate and plan without writing to the database. "
            "Still downloads the archive and queries existing records."
        ),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe all annotation tables and repopulate from the full archive",
    )
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()  # Keeps stdout alive for podman logs
        ],
    )

    # DB insertion logs are very noisy, so for this script we disable them
    logging.getLogger("sqlalchemy.engine").disabled = True
    logging.getLogger("sqlalchemy.engine.Engine").disabled = True

    try:
        asyncio.run(run_update(args.url, args.dry_run, args.reset))
    except Exception:  # pylint: disable=broad-exception-caught
        logger.exception("Update failed")
        sys.exit(1)
