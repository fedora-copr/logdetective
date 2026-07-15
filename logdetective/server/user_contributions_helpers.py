"""Helper functions for processing archive with user contributions from logdetective.com."""

import io
import json
import logging
import re
import tarfile
import tempfile
from contextlib import asynccontextmanager
from datetime import date
from typing import AsyncIterator, Iterator

import aiohttp
import backoff
from sqlalchemy import delete

from logdetective.server.database.base import (
    transaction,
)
from logdetective.server.database.models.annotated_builds import (
    AnnotatedBuilds,
    AnnotatedSnippets,
    AnnotationUpdates,
)
from logdetective.server.models import ContributionBuild

logger = logging.getLogger(__name__)


# Hard abort: kill the download if total response body exceeds this.
MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024
# SpooledTemporaryFile threshold: data stays in RAM below this, rolls to a
# temp file on disk above it.  The spool can still grow up to MAX_DOWNLOAD_BYTES.
SPOOL_ROLLOVER_BYTES = 50 * 1024 * 1024
DOWNLOAD_TIMEOUT = aiohttp.ClientTimeout(total=600)


class CurrentUpdateStats:
    """Stats accumulated during a single script run."""

    def __init__(self):
        self.jsons_processed = 0
        self.builds_added = 0
        self.snippets_added = 0


class ValidatedSnippet:
    """A snippet ready for DB insertion, with the log file it came from."""

    def __init__(self, log_name: str, text: str, user_comment: str):
        self.log_name = log_name
        self.text = text
        self.user_comment = user_comment


class ValidatedBuild:
    """A validated contribution ready for DB insertion."""

    def __init__(self, source_path: str, data: ContributionBuild):
        self.source_path = source_path  # relative path within archive
        self.fail_reason = data.fail_reason
        self.how_to_fix = data.how_to_fix
        self.snippets = [
            ValidatedSnippet(log_name, s.text, s.user_comment)
            for log_name, entry in data.logs.items()
            for s in entry.snippets
        ]


async def reset_tables() -> None:
    """Wipe all annotated build related data from the DB."""
    async with transaction(commit=True) as session:
        await session.execute(delete(AnnotatedSnippets))
        await session.execute(delete(AnnotatedBuilds))
        await session.execute(delete(AnnotationUpdates))
    logger.warning("All annotation tables wiped.")


def sanitize_for_pg(prefix: str, value: str) -> str:
    """Strip null bytes - the only byte PostgreSQL rejects in UTF-8 text columns."""
    if (null_pos := value.find("\x00")) != -1:
        logger.warning(
            "Stripping null char at position %d from %s: '%s'",
            null_pos, prefix, repr(value[max(0, null_pos - 20): null_pos + 20])
        )
        return value.replace("\x00", "")
    return value


@backoff.on_exception(
    backoff.expo,
    (aiohttp.ClientConnectionError, aiohttp.ServerTimeoutError, TimeoutError),
    max_tries=3,
)
async def _download_to_spool(
    spool: tempfile.SpooledTemporaryFile, url: str, params: dict[str, str],
) -> str | None:
    """
    Stream archive data into the provided spool.

    Returns the archive filename or None for empty archive.
    """
    spool.seek(0)
    spool.truncate()
    logger.info("GET %s params=%s", url, params)
    async with aiohttp.ClientSession(timeout=DOWNLOAD_TIMEOUT) as session:
        async with session.get(url, params=params) as resp:
            logger.info("Response: %d %s", resp.status, resp.reason)
            if resp.status == 204:
                logger.info("Server returned 204 No Content - no new contributions")
                return None
            resp.raise_for_status()

            disposition = resp.headers.get("Content-Disposition", "")
            logger.info("Content-Disposition: %s", disposition or "(not set)")
            filename = (
                disposition.split("filename=")[-1].strip('"')
                if "filename=" in disposition
                else ""
            )

            total = 0
            async for chunk in resp.content.iter_chunked(64 * 1024):
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    logger.error(
                        "Download exceeded size limit (%d bytes), aborting",
                        MAX_DOWNLOAD_BYTES,
                    )
                    raise RuntimeError(
                        f"Archive download exceeded {MAX_DOWNLOAD_BYTES} bytes, aborting."
                    )
                spool.write(chunk)
            spool.seek(0)
            logger.info("Downloaded %d bytes (%s)", total, filename)
            return filename


@asynccontextmanager
async def fetch_and_parse(
    url: str, stats: CurrentUpdateStats, since: date | None = None,
) -> AsyncIterator[tuple[Iterator[tuple[str, dict]], str]]:
    """
    Download and parse the contributions archive.

    If the server rejects the `since` parameter (400), falls back to
    downloading the full archive without `since`.

    Returns (map contribution_filename -> annotated build data, archive_filename).
    """
    params: dict[str, str] = {}
    if since:
        params["since"] = since.isoformat()

    logger.info("Downloading archive from %s (since=%s)", url, since)
    with tempfile.SpooledTemporaryFile(max_size=SPOOL_ROLLOVER_BYTES) as spool:
        try:
            filename = await _download_to_spool(spool, url, params)
        except aiohttp.ClientResponseError as exc:
            if exc.status == 400 and since:
                logger.warning(
                    "Server rejected since=%s (likely >90 days old), "
                    "falling back to full archive download.",
                    since,
                )
                spool.seek(0)
                spool.truncate()
                filename = await _download_to_spool(spool, url, {})
            else:
                raise

        if filename is None:
            yield iter(()), ""
            return

        logger.info("Parsing archive '%s'", filename)
        yield parse_archive(spool, stats), filename


def extract_archive_date(filename: str) -> date:
    """
    Extract the date from an archive filename. /download returns 'results-YYYY-MM-DD.tar.gz'
    and /download?since returns 'results-since-YYYY-MM-DD.tar.gz'.
    """
    match = re.search(r"results-(?:since-)?(?P<archive_date>\d{4}-\d{2}-\d{2})", filename)
    if not match:
        raise RuntimeError(
            f"Could not extract date from archive filename: {filename!r}"
        )
    return date.fromisoformat(match.group("archive_date"))


def _is_safe_tar_member(name: str) -> bool:
    """Reject tar member paths that could escape the archive root."""
    parts = name.split("/")
    return not (name.startswith("/") or ".." in parts)


def parse_archive(
    archive_file: io.IOBase,
    stats: CurrentUpdateStats,
) -> Iterator[tuple[str, dict]]:
    """
    Extract JSON contributions from a tar.gz archive file object.

    Returns a dict mapping relative file paths to parsed JSON data.
    """
    with tarfile.open(fileobj=archive_file, mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.name.startswith("results/results/"):
                continue
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            if not _is_safe_tar_member(member.name):
                logger.warning("Skipping unsafe tar member path: %s", member.name)
                continue
            try:
                f = tar.extractfile(member)
                if f is None:
                    logger.warning("Skipping %s: tar.extractfile returned None", member.name)
                    continue
                data = json.load(f)
                stats.jsons_processed += 1
                if not isinstance(data, dict):
                    logger.warning("Skipping %s: top-level JSON is not an object", member.name)
                    continue
                yield member.name, data
            except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
                logger.warning("Failed to parse %s: %s", member.name, exc)


async def log_update_report(current: CurrentUpdateStats, dry_run: bool) -> None:
    """Log a summary report of the current run and overall DB state."""
    build_count = await AnnotatedBuilds.get_count()
    snippet_count = await AnnotatedSnippets.get_count()

    skipped = current.jsons_processed - current.builds_added

    logger.info("--- Current update ---")
    logger.info("JSONs processed: %d", current.jsons_processed)
    logger.info("Builds %sadded: %d", "(to be) " if dry_run else "", current.builds_added)
    logger.info("Snippets %sadded: %d", "(to be) " if dry_run else "", current.snippets_added)
    if skipped > 0:
        logger.info("Skipped (duplicates, invalid, or malformed): %d", skipped)
    logger.info("--- Overall DB state ---")
    logger.info("Total annotated builds: %d", build_count)
    logger.info("Total annotated snippets: %d", snippet_count)
