import logging
import os
import re
import subprocess as sp
from typing import (
    Iterator,
    List,
    Dict,
    Tuple,
    Generator,
    Optional,
    NamedTuple,
)

from jinja2 import exceptions
import numpy as np
import yaml

from llama_cpp import (
    Llama,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
)
from logdetective.constants import SNIPPET_DELIMITER
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
            re.IGNORECASE
        ),
        "copr-team@redhat.com",
    ),
    (  # GPG fingerprints
        re.compile(
            r"\bFingerprint:\s*([0-9A-F]{32,64}|(?:\s*[0-9A-F]{4}){8,16})\b",
            re.IGNORECASE
        ),
        f"Fingerprint:{' FFFF' * 10}",
    ),
    (  # RSA keys, sometimes they are as short as 16 hexa characters
        re.compile(r"\bRSA\s+key\s+[0-9A-F]{16,512}\b", re.IGNORECASE),
        f"RSA key {'FFFF' * 10}"
    ),
    (  # Pubkeys, sometimes pubkey-deadbeef-01234567, or pubkey-40hexacharacters-other8
        re.compile(r"\bpubkey-[0-9A-F]{8}[0-9A-F-]{8,128}\b", re.IGNORECASE),
        f"pubkey-{'ffff' * 10}",
    )
]


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
    lines = text.splitlines()

    # Chunk we will be yielding
    chunk = ""
    # Number of line where the message started
    original_line = 0
    for i, line in enumerate(lines):
        if len(line) == 0:
            continue
        if new_message(line):
            # Yield chunk if we have it
            if len(chunk) > 0:
                yield (original_line, chunk)
            original_line = i
            chunk = line
        else:
            chunk += "\n" + line
        if len(chunk) > max_chunk_len:
            # If the chunk is too long, keep splitting into smaller chunks
            # until we reach manageable size
            while len(chunk) > max_chunk_len:
                remainder = chunk[max_chunk_len:]
                chunk = chunk[:max_chunk_len]
                yield (original_line, chunk)
                chunk = remainder

    # if we still have some text left over
    yield (original_line, chunk)


def initialize_model(
    model_pth: str, filename_suffix: str = ".gguf", verbose: bool = False
) -> Llama:
    """Initialize Llama class for inference.
    Args:
        model_pth (str): path to gguf model file or Hugging Face name
        filename_suffix (str): suffix of the model file name to be pulled from Hugging Face
        verbose (bool): level of verbosity for llamacpp
    """

    LOG.info("Loading model from %s", model_pth)

    if os.path.isfile(model_pth):
        model = Llama(
            model_path=model_pth,
            n_ctx=0,  # Maximum context for the model
            verbose=verbose,
            logits_all=True,
        )
    else:
        model = Llama.from_pretrained(
            model_pth,
            f"*{filename_suffix}",
            n_ctx=0,  # Maximum context for the model
            verbose=verbose,
            logits_all=True,
        )

    return model


def compute_certainty(probs: List[Dict]) -> float:
    """Compute certainty of repsponse based on average logit probability.
    Log probability is log(p), isn't really readable for most people, especially in compound.
    In this case it's just a matter of applying inverse operation exp.
    Of course that leaves you with a value in range <0, 1> so it needs to be multiplied by 100.
    Simply put, this is the most straightforward way to get the numbers out.

    This function is used in the server codebase.
    """

    top_logprobs = [np.exp(e["logprob"]) * 100 for e in probs]

    certainty = np.median(top_logprobs, axis=0)
    if np.isnan(certainty):
        raise ValueError("NaN certainty of answer")
    return certainty


def sanitize_log(log: str) -> str:
    """Redact personal identifiers from the log content before it is sent to the LLM for analysis.

    Redaction is done by replacing emails, and various public keys/signatures.
    """
    for pattern, replacement in SANITIZE_PATTERNS:
        log = re.sub(pattern, replacement, log)
    return log


def process_log(
    log: str,
    model: Llama,
    stream: bool,
    prompt_templates: PromptConfig | PromptManager,
    temperature: float,
) -> CreateChatCompletionResponse | Iterator[CreateChatCompletionStreamResponse]:
    """Processes a given log using the provided language model and returns its summary.

    Args:
        log (str): The input log to be processed.
        model (Llama): The language model used for processing the log.
        stream (bool): Return output as Iterator.
        prompt_templates (PromptConfig | PromptManager): Prompt templates to use with LLM.
        temperature (float): Temperature parameter for model runtime.
    Returns:
        str: The summary of the given log generated by the language model.
    """
    messages = [
        {"role": "system", "content": prompt_templates.default_system_prompt},
        {"role": "user", "content": prompt_templates.prompt_template.format(log)},
    ]

    response = model.create_chat_completion(
        messages=messages,
        stream=stream,
        max_tokens=0,
        logprobs=True,
        top_logprobs=1,
        temperature=temperature,
    )

    return response


def format_snippets(snippets: list[str] | list[Tuple[int, str]]) -> str:
    """Format snippets, giving them separator, id and finally
    concatenating them. If snippets have line number attached,
    include that in prompt.

    Line number must be first element in the tuple. Mixed format of snippets
    is permitted, but may have impact on inference.
    """
    summary = "\n"
    for i, s in enumerate(snippets):
        if isinstance(s, tuple):
            line_number, snippet_content = s
            header = f"Snippet No. {i} at line #{line_number}:"
        else:
            header = f"Snippet No. {i}:"
            snippet_content = s
        summary += f"{header}\n\n{snippet_content}\n{SNIPPET_DELIMITER}\n\n"
    return summary


def load_prompts(
    config_path: Optional[str] = None, template_path: Optional[str] = None
) -> PromptConfig | PromptManager:
    """Load prompts from yaml file, and optionally initialize `PromptManager`
    if provided with path to prompt templates.
    """
    configuration = PromptConfig()
    if config_path:
        try:
            with open(config_path, "r") as file:
                configuration = PromptConfig(**yaml.safe_load(file))
        except FileNotFoundError:
            LOG.error(
                "Prompt configuration file not found, reverting to defaults.",
                exc_info=True,
            )
    if template_path:
        try:
            return PromptManager(template_path, configuration)
        except exceptions.TemplateError:
            LOG.error(
                "Prompt templates couldn't be rendered, reverting to defaults.",
                exc_info=True,
            )
    return configuration


def prompt_to_messages(
    user_message: str,
    system_prompt: str | None = None,
    system_role: str = "developer",
    user_role: str = "user",
) -> List[Dict[str, str]]:
    """Turn prompt into list of message dictionaries.
    If `system_role` and `user_role` are the same, only a single message is created,
    as concatenation of `user_message` and `system_prompt`. This is useful for models which
    do not have separate system role, such as mistral.
    """

    if system_role == user_role:
        messages = [
            {"role": system_role, "content": f"{system_prompt}\n{user_message}"}
        ]
    else:
        messages = [
            {"role": system_role, "content": system_prompt},
            {
                "role": user_role,
                "content": user_message,
            },
        ]
    for m in messages:
        LOG.debug("'%s' prompt content: \n %s", m["role"], m["content"])
    return messages


def filter_snippet_patterns(snippet: str, skip_snippets: SkipSnippets) -> bool:
    """Try to match snippet agains provided patterns to determine if we should
    filter it out or not."""
    for key, pattern in skip_snippets.snippet_patterns.items():
        if pattern.match(snippet):
            LOG.debug("Snippet `%s` has matched against skip pattern %s", snippet, key)
            return True

    return False


def load_skip_snippet_patterns(path: str | None) -> SkipSnippets:
    """Load dictionary of snippet patterns we want to skip."""
    if path:
        try:
            with open(path, "r") as file:
                return SkipSnippets(yaml.safe_load(file))
        except OSError as e:
            LOG.error("Couldn't open file with snippet skip patterns `%s`", path)
            raise e

    return SkipSnippets({})


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
        result: the check is successful (content-size info is present within limits)
        value_present: content-length info is present (value_present=False => result=False)
        size_in_bytes: None if content-length missing or invalid
    """
    result: bool
    value_present: bool
    size_in_bytes: int | None


def check_content_size(
    headers: dict[str, str],
    size_limit: int
) -> ContentSizeCheck:
    """
    Validate that a request's content size doesn't exceed a maximum based on headers.

    Args:
        headers: Dictionary of HTTP headers

    Returns:
        ContentSizeCheck, If its `.result=False` => request should be rejected.
    """
    header_name = "Content-Length"
    content_length: str | int | None = (
        headers.get(header_name) or headers.get(header_name.lower())
    )

    if content_length is None:
        transfer_header = "Transfer-Encoding"
        transfer_encoding = (
            headers.get(transfer_header) or headers.get(transfer_header.lower(), "None")
        )
        LOG.warning(
            (
                "No `Content-Length`. Transfer-Encoding: %s "
                "Treating artifacts as over the maximum size."
            ),
            transfer_encoding
        )
        return ContentSizeCheck(result=False, value_present=False, size_in_bytes=None)

    try:
        size = int(content_length)
    except (ValueError, TypeError):
        LOG.error("Invalid Content-Length header value: %s", content_length)
        return ContentSizeCheck(result=False, value_present=True, size_in_bytes=None)

    is_valid = size <= size_limit
    if not is_valid:
        LOG.warning(
            "Content-Length: %d B (%.2f MiB) exceeds max %d B (%.2f MiB)",
            size, size / (1024 * 1024), size_limit, size_limit / (1024 * 1024),
        )
    return ContentSizeCheck(result=is_valid, value_present=True, size_in_bytes=size)


def mine_logs(log: str, extractors: list) -> List[Tuple[int, str]]:
    """Extract snippets from log text using extractors provided.
    Each extractor is applied in turn on original log.
    Depending on characteristics of extractors used, there may be
    an overlap in snippets extracted."""

    log_summary = []

    LOG.info("Getting summary")

    for extractor in extractors:
        log_summary.extend(extractor(log))

    ratio = len("\n".join([text for _, text in log_summary])) / len(log)
    LOG.debug("Log summary: \n %s", log_summary)
    LOG.info("Snippets: %s, Compression ratio: %4f", len(log_summary), ratio)

    return log_summary
