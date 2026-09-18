"""Artifact preparation shared by durable analysis workers."""

from __future__ import annotations

import aiohttp

from logdetective.config import LOG, SERVER_CONFIG
from logdetective.models import AnalysisRequest, ArtifactFile, RemoteArtifactFile
from logdetective.remote_log import RemoteLog
from logdetective.utils import run_blocking, sanitize_artifact


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
