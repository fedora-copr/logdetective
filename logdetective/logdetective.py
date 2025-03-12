import argparse
import logging
import sys

from logdetective.constants import DEFAULT_ADVISOR
from logdetective.utils import (
    process_log, initialize_model, retrieve_log_content, format_snippets, compute_certainty)
from logdetective.extractors import LLMExtractor, DrainExtractor

LOG = logging.getLogger("logdetective")


def setup_args():
    """ Setup argument parser and return arguments. """
    parser = argparse.ArgumentParser("logdetective")
    parser.add_argument("file", type=str,
                        default="", help="The URL or path to the log file to be analyzed.")
    parser.add_argument("-M", "--model",
                        help="The path or Hugging Face name of the language model for analysis.",
                        type=str, default=DEFAULT_ADVISOR)
    parser.add_argument("-F", "--filename_suffix",
                        help="Suffix of the model file name to be retrieved from Hugging Face.\
                            Makes sense only if the model is specified with Hugging Face name.",
                        default="Q4_K_S.gguf")
    parser.add_argument("-n", "--no-stream", action='store_true')
    parser.add_argument("-S", "--summarizer", type=str, default="drain",
                        help="Choose between LLM and Drain template miner as the log summarizer.\
                                LLM must be specified as path to a model, URL or local file.")
    parser.add_argument("-N", "--n_lines", type=int,
                        default=8, help="The number of lines per chunk for LLM analysis.\
                            This only makes sense when you are summarizing with LLM.")
    parser.add_argument("-C", "--n_clusters", type=int, default=8,
                        help="Number of clusters for Drain to organize log chunks into.\
                            This only makes sense when you are summarizing with Drain")
    parser.add_argument("-v", "--verbose", action='count', default=0)
    parser.add_argument("-q", "--quiet", action='store_true')
    return parser.parse_args()


def main():
    """Main execution function."""
    args = setup_args()

    if args.verbose and args.quiet:
        sys.stderr.write("Error: --quiet and --verbose is mutually exclusive.\n")
        sys.exit(2)

    # Logging facility setup
    log_level = logging.INFO
    if args.verbose >= 1:
        log_level = logging.DEBUG
    if args.quiet:
        log_level = 0

    logging.basicConfig(stream=sys.stdout)
    LOG.setLevel(log_level)

    # Primary model initialization
    try:
        model = initialize_model(args.model, filename_suffix=args.filename_suffix,
                                 verbose=args.verbose > 2)
    except ValueError as e:
        LOG.error(e)
        LOG.error("You likely do not have enough memory to load the AI model")
        sys.exit(3)

    # Log file summarizer selection and initialization
    if args.summarizer == "drain":
        extractor = DrainExtractor(args.verbose > 1, context=True, max_clusters=args.n_clusters)
    else:
        summarizer_model = initialize_model(args.summarizer, verbose=args.verbose > 2)
        extractor = LLMExtractor(summarizer_model, args.verbose > 1)

    LOG.info("Getting summary")

    try:
        log = retrieve_log_content(args.file)
    except ValueError as e:
        # file does not exists
        LOG.error(e)
        sys.exit(4)
    log_summary = extractor(log)

    ratio = len(log_summary) / len(log.split('\n'))

    LOG.info("Compression ratio: %s", ratio)

    LOG.info("Analyzing the text")

    log_summary = format_snippets(log_summary)
    LOG.info("Log summary: \n %s", log_summary)

    stream = True
    if args.no_stream:
        stream = False
    response = process_log(log_summary, model, stream)
    probs = []
    print("Explanation:")
    # We need to extract top token probability from the response
    # CreateCompletionResponse structure of llama-cpp-python.
    # `compute_certainty` function expects list of dictionaries with form
    # { 'logprob': <float> } as expected from the OpenAI API.

    if args.no_stream:
        print(response["choices"][0]["text"])
        probs = [{'logprob': e} for e in response['choices'][0]['logprobs']['token_logprobs']]

    else:
        # Stream the output
        for chunk in response:
            if isinstance(chunk["choices"][0]["logprobs"], dict):
                probs.append({'logprob': chunk["choices"][0]["logprobs"]['token_logprobs'][0]})
            delta = chunk['choices'][0]['text']
            print(delta, end='', flush=True)
    certainty = compute_certainty(probs)

    print(f"\nResponse certainty: {certainty:.2f}%\n")


if __name__ == "__main__":
    main()
