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

## Cancellation ownership

Shared build-agent setup and execution live in `logdetective.agents.common`.
Each attempt owns its abort controller and nested blocking-work scope. Task
cancellation signals BeeAI and joins running work before terminal reporting.
Repeated cancellation does not abandon threads or clear cancellation counts;
an enclosing timeout removes only its own cancellation request. Running threads
cannot be forcibly stopped, so cleanup can wait indefinitely for a stuck call.
Queue reconciliation confirms cancellation even when the queue job fails and
never overwrites an accepted cancellation with queue failure.
