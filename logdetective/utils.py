import logging
import re
import subprocess as sp
from collections.abc import Mapping
from typing import (
    List,
    Tuple,
    Generator,
    Optional,
    NamedTuple,
)

import tomllib
import yaml

from logdetective.constants import TRUNCATED, MINIMUM_SNIPPET_TRUNCATION_LEN
from logdetective.models import PromptConfig, SkipSnippets
from logdetective.prompts import PromptManager


LOG = logging.getLogger("logdetective")

SANITIZE_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (  # Emails
        # we don't want to match invalid subdomains, starting/ending with - or .
        # such as @-domain.com or @domain-.com or @.domain.com or @domain..com
        re.compile(
            (
                r"\b[\w.%+-]+"  # username
                r"(?:@|\(at\)|\[at\])"  # "at" symbol
                r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.){1,10}"  # subdomains
                r"[a-z]{2,4}\b"  # top level domain
            ),
            re.IGNORECASE,
        ),
        "copr-team@redhat.com",
    ),
    (  # GPG fingerprints
        re.compile(
            r"\bFingerprint:\s*([0-9A-F]{32,64}|(?:\s*[0-9A-F]{4}){8,16})\b",
            re.IGNORECASE,
        ),
        f"Fingerprint:{' FFFF' * 10}",
    ),
    (  # RSA keys, sometimes they are as short as 16 hexa characters
        re.compile(r"\bRSA\s+key\s+[0-9A-F]{16,512}\b", re.IGNORECASE),
        f"RSA key {'FFFF' * 10}",
    ),
    (  # Pubkeys, sometimes pubkey-deadbeef-01234567, or pubkey-40hexacharacters-other8
        re.compile(r"\bpubkey-[0-9A-F]{8}[0-9A-F-]{8,128}\b", re.IGNORECASE),
        f"pubkey-{'ffff' * 10}",
    ),
]


# pylint: disable=missing-function-docstring
def mib_to_bytes(mib: int) -> int:
    return mib * 1024 * 1024


def new_message(text: str) -> bool:
    """Set of heuristics for determining whether or not
    does the current chunk of log text continue on next line.

    Following rules are checked, in order:
    * is the first character is whitespace
    * is the first character backslash '|'
    """
    conditionals = [
        lambda string: string[0].isspace(),
        lambda string: string[0] == "|",
    ]

    for c in conditionals:
        y = c(text)
        if y:
            return False

    return True


def get_chunks(
    text: str, max_chunk_len: int = 2000
) -> Generator[Tuple[int, str], None, None]:
    """Split log into chunks according to heuristic
    based on whitespace and backslash presence.
    """
    if max_chunk_len < MINIMUM_SNIPPET_TRUNCATION_LEN:
        raise ValueError(f"Snippets must be at least {MINIMUM_SNIPPET_TRUNCATION_LEN} chars long")
    lines = text.splitlines()

    # Chunk we will be yielding
    chunk = ""
    # Number of line where the message started
    original_line = 1
    for i, line in enumerate(lines, start=1):
        if len(line) == 0:
            continue
        if new_message(line):
            # Yield chunk if we have it
            if len(chunk) > 0:
                yield (original_line, chunk)
            original_line = i
            chunk = line
        else:
            # If the chunk was truncated, we'll start building a new one
            if not chunk:
                original_line = i
                chunk = line
            else:
                chunk += "\n" + line
        if len(chunk) > max_chunk_len:
            # If the chunk is too long, truncate and add <truncated> tag
            chunk = chunk[:max(max_chunk_len - len(TRUNCATED), 0)] + TRUNCATED
            yield (original_line, chunk)
            chunk = ""
    # if we still have some text left over
    if chunk:
        yield (original_line, chunk)


def sanitize_artifact(log: str) -> str:
    """Redact personal identifiers from the artifact content before it is sent to the LLM.

    Redaction is done by replacing emails, and various public keys/signatures.
    """
    for pattern, replacement in SANITIZE_PATTERNS:
        log = re.sub(pattern, replacement, log)
    return log


def load_prompts(
    template_path: str, config_path: Optional[str] = None
) -> PromptManager:
    """Load prompt templates from given path, optionally load PromptConfig
    and initialize `PromptManager`.
    If templates are missing or malformed raise an exception."""

    configuration = PromptConfig()
    if config_path:
        try:
            with open(config_path, "r") as file:
                configuration = PromptConfig(**yaml.safe_load(file))
        except (FileNotFoundError, TypeError):
            LOG.error(
                "Prompt configuration file not found or empty, reverting to defaults.",
                exc_info=True,
            )

    return PromptManager(template_path, configuration)


def filter_snippet_patterns(
    snippet: str,
    filename: str | None = None,
    skip_snippets: Optional[SkipSnippets] = None,
) -> bool:
    """Try to match snippet against provided patterns to determine if we should
    filter it out or not."""
    if not skip_snippets:
        return False
    for key, pattern in skip_snippets.get_patterns_for_file(filename).items():
        if pattern.match(snippet):
            LOG.debug("Snippet `%s` has matched against skip pattern %s", snippet, key)
            return True

    return False


def load_skip_snippet_patterns(path: str | None) -> SkipSnippets | None:
    """Load dictionary of snippet patterns we want to skip."""
    if path:
        try:
            with open(path, "rb") as file:
                data = tomllib.load(file)
            if not data:
                LOG.warning("Skip pattern file `%s` is empty, no skip patterns loaded", path)
                return None
            return SkipSnippets(data)
        except FileNotFoundError:
            LOG.error(
                "Couldn't open file with snippet skip patterns `%s`",
                path,
                stack_info=True,
            )
    return None


def check_csgrep() -> bool:
    """Verifies presence of csgrep in path"""
    try:
        result = sp.run(
            ["csgrep", "--version"],
            text=True,
            check=True,
            shell=False,
            capture_output=True,
            timeout=1.0,
        )
    except (FileNotFoundError, sp.TimeoutExpired, sp.CalledProcessError) as ex:
        LOG.error("Required binary `csgrep` was not found in path: %s", ex)
        return False
    if result.returncode == 0:
        return True
    LOG.error("Issue was encountered while calling `csgrep`: `%s`", result.stderr)

    return False


class ContentSizeCheck(NamedTuple):
    """
    Aggregate requests content-size info for checks.

    Args:
        proceed: True if download should proceed; False to reject.
        size_in_bytes: Parsed Content-Length value, or None if absent or invalid.
    """

    proceed: bool
    size_in_bytes: int | None


def check_content_size(
    headers: Mapping,
    size_limit: int,
    require_header: bool = True,
) -> ContentSizeCheck:
    """
    Validate that a request's content size doesn't exceed a maximum based on headers.

    Args:
        headers: HTTP headers
        size_limit: Maximum allowed size in bytes
        require_header:
            True(defualt): request with no Content-Length header is rejected.
            False: request with an absent header is allowed (limit is then checked while reading).

    Returns:
        ContentSizeCheck.`.proceed=True` means safe to download. `.proceed=False` means reject.
        `.size_in_bytes` is None when the header is absent or unparseable.
    """
    content_length = headers.get("content-length")

    if content_length is None:
        transfer_encoding = headers.get("transfer-encoding", "None")
        LOG.info(
            "No Content-Length header. Transfer-Encoding: %s",
            transfer_encoding,
        )
        return ContentSizeCheck(proceed=not require_header, size_in_bytes=None)

    try:
        size = int(content_length)
    except (ValueError, TypeError):
        LOG.error("Invalid Content-Length header value: %s", content_length)
        return ContentSizeCheck(proceed=False, size_in_bytes=None)

    is_valid = size <= size_limit
    if not is_valid:
        LOG.warning(
            "Content-Length: %d B (%.2f MiB) exceeds max %d B (%.2f MiB)",
            size,
            size / (1024 * 1024),
            size_limit,
            size_limit / (1024 * 1024),
        )
    return ContentSizeCheck(proceed=is_valid, size_in_bytes=size)
