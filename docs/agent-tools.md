# Agent tools

The agent must call `drain_extractor` once for every build artifact before
using optional extractors or providing the final answer. Other extractors do
not satisfy this requirement. When available, `csgrep_extractor` captures GCC
compiler traces and `python_traceback_extractor` captures Python exception
tracebacks. The start prompt mentions each optional tool only when the current
agent run offers it. BeeAI's Drain invocation requirement remains the hard
gate; prompt ordering is guidance to reduce invalid attempts.

When the annotation database contains examples and an embedding model is
available, `annotated_snippet_lookup` can search for similar failures. The agent
passes a specific extracted error as `snippet`. Returned annotations, build
problems, and solutions describe past builds.

`snippet_annotation` attaches a brief description to a useful extracted snippet.
Its `source_file` and `line_number` must match extractor output. The description
goes in the `snippet_analysis` argument.
