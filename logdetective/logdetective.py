import argparse
import logging
import os
import sys

from logdetective.constants import DEFAULT_ADVISOR, CACHE_LOC
from logdetective.utils import download_model, process_log, initialize_model, retrieve_log_content
from logdetective.extractors import LLMExtractor, DrainExtractor

LOG = logging.getLogger("logdetective")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser("logdetective")
    parser.add_argument("file", type=str,
                        default="", help="The URL or path to the log file to be analyzed.")
    parser.add_argument("-M", "--model", help="The path or URL of the language model for analysis.",
                        type=str, default=DEFAULT_ADVISOR)
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
    args = parser.parse_args()

    if args.verbose and args.quiet:
        sys.stderr.write("Error: --quiet and --verbose is mutually exclusive.\n")
        sys.exit(2)
    log_level = logging.INFO
    if args.verbose >= 1:
        log_level = logging.DEBUG
    if args.quiet:
        log_level = 0
    logging.basicConfig(stream=sys.stdout)
    LOG.setLevel(log_level)

    if not os.path.exists(CACHE_LOC):
        os.makedirs(os.path.expanduser(CACHE_LOC), exist_ok=True)

    if not os.path.isfile(args.model):
        model_pth = download_model(args.model, not args.quiet)
    else:
        model_pth = args.model

    if args.summarizer == "drain":
        extractor = DrainExtractor(args.verbose > 1, context=True, max_clusters=args.n_clusters)
    elif os.path.isfile(args.summarizer):
        extractor = LLMExtractor(args.summarizer, args.verbose > 1, args.n_lines)
    else:
        summarizer_pth = download_model(args.summarizer, not args.quiet)
        extractor = LLMExtractor(summarizer_pth, args.verbose > 1)

    LOG.info("Getting summary")
    model = initialize_model(model_pth, args.verbose > 2)

    log = retrieve_log_content(args.file)
    log_summary = extractor(log)

    ratio = len(log_summary.split('\n')) / len(log.split('\n'))
    LOG.debug("Log summary: \n %s", log_summary)
    LOG.info("Compression ratio: %s", ratio)

    LOG.info("Analyzing the text")
    print(f"Explanation: \n{process_log(log_summary, model)}")


if __name__ == "__main__":
    main()
