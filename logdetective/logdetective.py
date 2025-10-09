import argparse
import asyncio
import logging
import sys
import os

import aiohttp

from logdetective.constants import DEFAULT_ADVISOR, DEFAULT_TEMPERATURE
from logdetective.utils import (
    process_log,
    initialize_model,
    retrieve_log_content,
    format_snippets,
    compute_certainty,
    load_prompts,
    load_skip_snippet_patterns,
    check_csgrep,
    mine_logs,
)
from logdetective.extractors import DrainExtractor, CSGrepExtractor

LOG = logging.getLogger("logdetective")


def setup_args():
    """Setup argument parser and return arguments."""
    parser = argparse.ArgumentParser("logdetective")
    parser.add_argument(
        "file",
        type=str,
        default="",
        help="The URL or path to the log file to be analyzed.",
    )
    parser.add_argument(
        "-M",
        "--model",
        help="The path or Hugging Face name of the language model for analysis.",
        type=str,
        default=DEFAULT_ADVISOR,
    )
    parser.add_argument(
        "-F",
        "--filename_suffix",
        help="Suffix of the model file name to be retrieved from Hugging Face.\
                            Makes sense only if the model is specified with Hugging Face name.",
        default="Q4_K.gguf",
    )
    parser.add_argument("-n", "--no-stream", action="store_true")
    parser.add_argument(
        "-S",
        "--summarizer",
        type=str,
        default="drain",
        help="DISABLED: LLM summarization option was removed. \
                Argument is kept for backward compatibility only.",
    )
    parser.add_argument(
        "-N",
        "--n_lines",
        type=int,
        default=None,
        help="DISABLED: LLM summarization option was removed. \
                Argument is kept for backward compatibility only.",
    )
    parser.add_argument(
        "-C",
        "--n_clusters",
        type=int,
        default=8,
        help="Number of clusters for Drain to organize log chunks into.\
                            This only makes sense when you are summarizing with Drain",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument(
        "--prompts",
        type=str,
        default=f"{os.path.dirname(__file__)}/prompts.yml",
        help="Path to prompt configuration file.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Temperature for inference.",
    )
    parser.add_argument(
        "--skip_snippets",
        type=str,
        default=f"{os.path.dirname(__file__)}/skip_snippets.yml",
        help="Path to patterns for skipping snippets.",
    )
    parser.add_argument(
        "--csgrep", action="store_true", help="Use csgrep to process the log."
    )
    return parser.parse_args()


async def run():  # pylint: disable=too-many-statements,too-many-locals,too-many-branches
    """Main execution function."""
    args = setup_args()

    if args.verbose and args.quiet:
        sys.stderr.write("Error: --quiet and --verbose is mutually exclusive.\n")
        sys.exit(2)

    # Emit warning about use of discontinued args
    if args.n_lines or args.summarizer != "drain":
        LOG.warning("LLM based summarization was removed. Drain will be used instead.")

    # Logging facility setup
    log_level = logging.INFO
    if args.verbose >= 1:
        log_level = logging.DEBUG
    if args.quiet:
        log_level = 0

    # Get prompts configuration
    prompts_configuration = load_prompts(args.prompts)

    logging.basicConfig(stream=sys.stdout)
    LOG.setLevel(log_level)

    # Primary model initialization
    try:
        model = initialize_model(
            args.model, filename_suffix=args.filename_suffix, verbose=args.verbose > 2
        )
    except ValueError as e:
        LOG.error(e)
        LOG.error("You likely do not have enough memory to load the AI model")
        sys.exit(3)

    try:
        skip_snippets = load_skip_snippet_patterns(args.skip_snippets)
    except OSError as e:
        LOG.error(e)
        sys.exit(5)

    # Log file summarizer initialization
    extractors = []
    extractors.append(
        DrainExtractor(
            args.verbose > 1,
            max_clusters=args.n_clusters,
            skip_snippets=skip_snippets,
        )
    )

    if args.csgrep:
        if not check_csgrep():
            LOG.error(
                "You have requested use of `csgrep` when it isn't available on your system."
            )
            sys.exit(6)
        extractors.append(
            CSGrepExtractor(args.verbose > 1, skip_snippets=skip_snippets)
        )

    LOG.info("Getting summary")

    async with aiohttp.ClientSession() as http:
        try:
            log = await retrieve_log_content(http, args.file)
        except ValueError as e:
            # file does not exist
            LOG.error(e)
            sys.exit(4)

    log_summary = mine_logs(log=log, extractors=extractors)
    LOG.info("Analyzing the text")

    log_summary = format_snippets(log_summary)
    LOG.info("Log summary: \n %s", log_summary)

    prompt = (
        f"{prompts_configuration.default_system_prompt}\n"
        f"{prompts_configuration.prompt_template}"
    )

    stream = True
    if args.no_stream:
        stream = False
    response = process_log(
        log_summary,
        model,
        stream,
        prompt_template=prompt,
        temperature=args.temperature,
    )
    probs = []
    print("Explanation:")
    # We need to extract top token probability from the response
    # CreateCompletionResponse structure of llama-cpp-python.
    # `compute_certainty` function expects list of dictionaries with form
    # { 'logprob': <float> } as expected from the OpenAI API.

    if args.no_stream:
        print(response["choices"][0]["text"])
        probs = [
            {"logprob": e} for e in response["choices"][0]["logprobs"]["token_logprobs"]
        ]

    else:
        # Stream the output
        for chunk in response:
            if isinstance(chunk["choices"][0]["logprobs"], dict):
                probs.append(
                    {"logprob": chunk["choices"][0]["logprobs"]["token_logprobs"][0]}
                )
            delta = chunk["choices"][0]["text"]
            print(delta, end="", flush=True)
    certainty = compute_certainty(probs)

    print(f"\nResponse certainty: {certainty:.2f}%\n")


def main():
    """Evaluate logdetective program and wait for it to finish"""
    asyncio.run(run())


if __name__ == "__main__":
    main()
