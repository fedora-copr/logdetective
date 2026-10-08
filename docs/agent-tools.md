# Agent tools

The agent extracts snippets from every build artifact with `drain_extractor`.
When available, `csgrep_extractor` captures GCC compiler traces and
`python_traceback_extractor` captures Python exception tracebacks. The start
prompt mentions each optional tool only when the current agent run offers it.

When the annotation database contains examples and an embedding model is
available, `annotated_snippet_lookup` can search for similar failures. The agent
passes a specific extracted error as `snippet`. Returned annotations, build
problems, and solutions describe past builds.

`snippet_annotation` attaches a brief description to a useful extracted snippet.
Its `source_file` and `line_number` must match extractor output. The description
goes in the `snippet_analysis` argument.
