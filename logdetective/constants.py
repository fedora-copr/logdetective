"""This file contains various constants to be used as a fallback
in case other values are not specified. Prompt templates should be modified
in prompts.yaml instead.
"""

# pylint: disable=line-too-long
DEFAULT_ADVISOR = "fedora-copr/Mistral-7B-Instruct-v0.2-GGUF"

PROMPT_TEMPLATE = """
Given following log snippets, and nothing else, explain what failure, if any, occured during build of this package.

Analysis of the snippets must be in a format of [X] : [Y], where [X] is a log snippet, and [Y] is the explanation.
Snippets themselves must not be altered in any way whatsoever.

Snippets are delimited with '================'.

Finally, drawing on information from all snippets, provide complete explanation of the issue and recommend solution.

Explanation of the issue, and recommended solution, should take handful of sentences.

Snippets:

{}

Analysis:

"""

SNIPPET_PROMPT_TEMPLATE = """
Analyse following RPM build log snippet. Describe contents accurately, without speculation or suggestions for resolution.

Your analysis must be as concise as possible, while keeping relevant information intact.

Snippet:

{}

Analysis:

"""

PROMPT_TEMPLATE_STAGED = """
Given following log snippets, their explanation, and nothing else, explain what failure, if any, occured during build of this package.

Snippets are in a format of [X] : [Y], where [X] is a log snippet, and [Y] is the explanation.

Snippets are delimited with '================'.

Drawing on information from all snippets, provide complete explanation of the issue and recommend solution.

Explanation of the issue, and recommended solution, should take handful of sentences.

Snippets:

{}

Analysis:

"""

DEFAULT_SYSTEM_PROMPT = """
You are a highly capable large language model based expert system specialized in
packaging and delivery of software using RPM (RPM Package Manager). Your purpose is to diagnose
RPM build failures, identifying root causes and proposing solutions if possible.
You are truthful, concise, and helpful.

You never speculate about package being built or fabricate information.
If you do not know the answer, you acknowledge the fact and end your response.
Your responses must be as short as possible.
"""

SNIPPET_DELIMITER = "================"

DEFAULT_TEMPERATURE = 0.8

# Tuning for LLM-as-a-Service
LLM_DEFAULT_MAX_QUEUE_SIZE = 50
LLM_DEFAULT_REQUESTS_PER_MINUTE = 60

# Roles for chat API
SYSTEM_ROLE_DEFAULT = "developer"
USER_ROLE_DEFAULT = "user"
