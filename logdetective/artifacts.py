"""Artifact preparation shared by durable analysis workers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import aiohttp

from logdetective.config import LOG, SERVER_CONFIG
from logdetective.exceptions import RemoteLogError, RemoteLogTooLargeError
from logdetective.models import (
    AnalysisRequest, ArtifactFile, RemoteArtifactFile,
    TaskError, TestingFarmCase, TestingFarmCaseOutcome, TestingFarmSuiteRequest,
    TestingFarmRetrievalWarning,
)
from logdetective.remote_log import RemoteLog
from logdetective.utils import (
    SSRFProtectedConnector, SSRFProtectedResolver, run_blocking, sanitize_artifact,
)


@dataclass
class SuiteArtifacts:
    """Downloaded suite logs and cases eligible for agent analysis."""

    artifacts: dict[str, str]
    eligible: list[TestingFarmCase]
    errors: dict[str, TestingFarmCaseOutcome]
    warnings: list[TestingFarmRetrievalWarning]


async def fetch_suite_log(
    url: str, http: aiohttp.ClientSession, remaining: int,
    fetched: dict[str, tuple[str | None, int]],
) -> tuple[str, int]:
    """Cache URL success/failure and account for bytes even on failed reads.

    Each download is limited by the suite's remaining aggregate byte allowance,
    not a separate per-log cap. A declared or streamed size exceeding that
    allowance raises RemoteLogTooLargeError and fails the whole suite.
    """
    if url in fetched:
        content, _ = fetched[url]
        if content is None:
            raise RemoteLogError("Cached retrieval failure")
        return content, 0
    remote = RemoteLog(url, http, limit_bytes=remaining)
    try:
        content = await remote.get_url_content()
        if remote.remote_log_size > remaining:
            raise RemoteLogTooLargeError("Aggregate artifact budget exceeded")
        if not content.strip():
            raise RemoteLogError("Empty log")
    except (RemoteLogError, aiohttp.ClientError, asyncio.TimeoutError):
        fetched[url] = (None, remote.remote_log_size)
        raise
    fetched[url] = (content, remote.remote_log_size)
    return content, remote.remote_log_size


# pylint: disable=too-many-locals
async def fetch_suite_artifacts(
    payload: TestingFarmSuiteRequest, request_size: int,
) -> SuiteArtifacts:
    """Fetch suite logs once, retaining non-budget failures as scoped warnings.

    aiohttp transport failures are handled here because RemoteLog may raise
    them directly during HEAD, GET, or response-body reads.
    Size-budget violations instead fail the entire suite, including when one
    log exceeds the remaining aggregate allowance. Bytes read by unsuccessful
    downloads also consume that allowance.
    """
    payload.validate_limits(SERVER_CONFIG.testing_farm)
    remaining = SERVER_CONFIG.general.max_artifact_size - request_size
    if remaining <= 0:
        raise ValueError("Total submitted artifact size exceeds the limit")
    artifacts: dict[str, str] = {}
    errors: dict[str, TestingFarmCaseOutcome] = {}
    eligible: list[TestingFarmCase] = []
    fetched: dict[str, tuple[str | None, int]] = {}
    warnings: list[TestingFarmRetrievalWarning] = []

    async def retrieve(log: RemoteArtifactFile, source_file: str, http):
        """Record non-budget failures as scoped warnings; propagate size violations."""
        nonlocal remaining
        url = str(log.url)
        cached = url in fetched
        try:
            content, consumed = await fetch_suite_log(url, http, remaining, fetched)
            remaining -= consumed
            return content
        except RemoteLogTooLargeError:
            # The download limit is the remaining suite budget, not a per-log cap.
            raise
        except (RemoteLogError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if not cached:
                remaining -= fetched.get(url, (None, 0))[1]
            if remaining < 0:
                raise RemoteLogTooLargeError("Aggregate artifact budget exceeded") from exc
            warnings.append(TestingFarmRetrievalWarning(
                code="artifact_unavailable", source_file=source_file,
                message="Unable to retrieve this artifact",
            ))
            return None
    ssrf_config = SERVER_CONFIG.general.ssrf_protection
    connector = (
        SSRFProtectedConnector(SSRFProtectedResolver(
            allowed_hostnames=ssrf_config.allowed_hostnames,
            blocked_cluster_cidrs=ssrf_config.blocked_cluster_cidrs,
        ))
        if SERVER_CONFIG.general.block_localhost_urls else aiohttp.TCPConnector()
    )
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(
            total=SERVER_CONFIG.general.log_source_request_timeout, connect=3.07
        ),
    ) as http:
        for log in payload.shared_artifacts:
            source_file = f"shared/{log.name}"
            content = await retrieve(log, source_file, http)
            if content is not None:
                artifacts[f"shared/{log.name}"] = content
        shared_available = any(
            f"shared/{log.name}" in artifacts for log in payload.shared_artifacts
        )
        for case in payload.cases:
            for log in case.artifacts:
                source_file = f"case/{case.case_id}/{log.name}"
                content = await retrieve(log, source_file, http)
                if content is not None:
                    artifacts[f"case/{case.case_id}/{log.name}"] = content
            usable = shared_available or any(
                f"case/{case.case_id}/{log.name}" in artifacts for log in case.artifacts
            )
            if not usable:
                errors[case.case_id] = TestingFarmCaseOutcome(
                    status="error", error=TaskError(
                        code="artifact_error", message="Unable to retrieve a case artifact"
                    ),
                )
            else:
                eligible.append(case)
    return SuiteArtifacts(artifacts=artifacts, eligible=eligible, errors=errors,
                          warnings=warnings)


async def get_artifacts_from_payload(
    payload: AnalysisRequest,
    http_session: aiohttp.ClientSession,
    request_size: int,
) -> dict[str, str | RemoteLog]:
    """Build bounded, sanitized artifacts from one validated request.

    Args:
        payload: Validated generic analysis request containing local or remote files.
        http_session: Session used to retrieve remote artifacts.
        request_size: Size in bytes of the submitted request body. Remote downloads
            share the remaining configured artifact-size allowance.

    Returns:
        A mapping from artifact names to sanitized text or lazily fetched remote logs.

    Raises:
        ValueError: If the submitted data leaves no capacity for a remote artifact.
        TypeError: If the request contains an unsupported artifact model.
        RemoteLogError: If eager retrieval of a remote artifact fails.
    """
    build_artifacts: dict[str, str | RemoteLog] = {}
    total_payload_size = request_size

    for artifact in payload.files:
        if isinstance(artifact, RemoteArtifactFile):
            remaining_limit = SERVER_CONFIG.general.max_artifact_size - total_payload_size
            if remaining_limit <= 0:
                raise ValueError("Total submitted artifact size exceeds the limit")
            remote_log = RemoteLog(
                str(artifact.url), http_session, limit_bytes=remaining_limit
            )
            if SERVER_CONFIG.general.delay_artifact_download:
                build_artifacts[artifact.name] = remote_log
            else:
                build_artifacts[artifact.name] = await remote_log.get_url_content()
                total_payload_size += remote_log.remote_log_size
        elif isinstance(artifact, ArtifactFile):
            build_artifacts[artifact.name] = await run_blocking(
                sanitize_artifact, artifact.content
            )
        else:
            raise TypeError(f"Unsupported artifact type {type(artifact)!r}")

    LOG.info("Prepared %d artifacts for durable analysis", len(build_artifacts))
    return build_artifacts
