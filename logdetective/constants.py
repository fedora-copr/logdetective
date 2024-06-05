
# pylint: disable=line-too-long
DEFAULT_ADVISOR = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"

PROMPT_TEMPLATE = """
Given following log snippets, and nothing else, explain what failure, if any, occured during build of this package.

{}

Analysis of the failure must be in a format of [X] : [Y], where [X] is a log snippet, and [Y] is the explanation.

Finally, drawing on information from all snippets, provide complete explanation of the issue.

Analysis:

"""

SUMMARIZE_PROMPT_TEMPLATE = """
Does following log contain error or issue?

Log:

{}

Answer:

"""
