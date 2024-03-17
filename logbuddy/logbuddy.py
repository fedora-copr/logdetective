import requests
import argparse
import os
from llama_cpp import Llama, LlamaGrammar
import numpy as np
import re
from typing import Union
from urllib.request import urlretrieve

DEFAULT_ADVISOR = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_S.gguf?download=true"

DEFAULT_LLM_RATER = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_S.gguf?download=true"

PROMPT_TEMPLATE = """
Following is a summary of a log file from an RPM build.
Identify reason for build failure, if any, and recommend solution:

{}

Reason for build failure and proposed solution:

"""

SUMMARIZE_PROPT_TEMPLATE = """
Does following log contain error or issue?

Log:

{}

Answer:

"""

BAD_STRINGS = "error warning fail missing dump fault"

CACHE_LOC = "~/.cache/logbuddy/"

class RegexRater:

    def __init__(self):
        pattern = BAD_STRINGS.split()
        pattern = f".*({'|'.join(pattern)}).*"
        self.pattern = re.compile(pattern, re.IGNORECASE)

    def __call__(self, log: str) -> str:
        if re.match(self.pattern, log):
            return "Yes"
        return "No"


class LLMRater:

    def __init__(self, model_path: str):
        self.model = Llama(
            model_path=model_path,
            n_ctx=0,
            verbose=False)
        self.grammar = LlamaGrammar.from_string(
            "root ::= (\"Yes\" | \"No\")", verbose=False)

    def __call__(self, log: str) -> str:
        prompt = SUMMARIZE_PROPT_TEMPLATE.format(log)
        out = self.model(prompt, max_tokens=7, grammar=self.grammar)
        out = f"{out['choices'][0]['text']}\n"


def download_model(url: str) -> None:
    path = os.path.join(
        os.path.expanduser(CACHE_LOC), url.split('/')[-1])

    if not os.path.exists(path):
        file, status = urlretrieve(url, path)

    return path


def rate_chunks(
        log: str, model: Union[LLMRater, RegexRater],
        n_lines: int = 10) -> list[tuple]:

    results = []
    log_lines = log.split("\n")

    for i in range(0, len(log_lines), n_lines):
        block = '\n'.join(log_lines[i:i+n_lines])
        out = model(block)
        results.append((block, out))

    return results


def create_extract(chunks: list[str], neighbors: bool = False) -> str:

    interesting = []
    summary = ""
    for i in range(len(chunks)):
        if chunks[i][1].startswith("Yes"):
            interesting.append(i)
            if neighbors:
                interesting.extend([max(i-1, 0), min(i+1, len(chunks)-1)])

    interesting = np.unique(interesting)

    for i in interesting:
        summary += chunks[i][0] + "\n"

    return summary


def process_log(log: str, model: Llama) -> str:
    return model(PROMPT_TEMPLATE.format(log), max_tokens=0)["choices"][0]["text"]


def main():
    parser = argparse.ArgumentParser("logbuddy")
    parser.add_argument("url", type=str, default="")
    parser.add_argument("-M", "--model", type=str, default=DEFAULT_ADVISOR)
    parser.add_argument("-S", "--summarizer", type=str, default="regex")
    args = parser.parse_args()

    if not os.path.exists(CACHE_LOC):
        os.makedirs(os.path.expanduser(CACHE_LOC), exist_ok=True)

    if not os.path.isfile(args.model):
        model_pth = download_model(args.model)
    else:
        model_pth = args.model

    if args.summarizer == "regex":
        rater = RegexRater()
    elif os.path.isfile(args.summarizer):
        rater = LLMRater(args.summarizer)
    else:
        summarizer_pth = download_model(args.summarizer)
        rater = LLMRater(summarizer_pth)

    model = Llama(
        model_path=model_pth,
        n_ctx=0,
        verbose=False)

    log = requests.get(args.url).text

    chunks = rate_chunks(log, rater)
    log_summary = create_extract(chunks, neighbors=True)

    ratio = len(log_summary.split('\n'))/len(log.split('\n'))

    print(f"Log summary: \n{log_summary}")
    print(f"Compression ratio: {ratio}")
    print(f"Explanation: \n{process_log(log_summary, model)}")


if __name__ == "__main__":
    main()
