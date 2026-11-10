# Agent tools

Agent runs translate task cancellation into BeeAI abort signals and join the
run before returning. A task-local blocking-work scope follows nested tool
contexts, waits for their already-started threads, and prevents detached tool
contexts from starting new blocking work after the run closes. This applies
to both timeout and explicit cancellation, including repeated cancellation.

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

Build and Testing Farm agents have separate workflows. Shared extractor setup
and result utilities live in `logdetective.agents.common`; the suite workflow
pre-extracts shared and case evidence before one whole-suite agent run.
The shared setup registers each tool with its call requirements and tracks
extractors in the same order used for snippet annotation and final responses.

For `tft_suite` tasks, the worker runs Drain extraction for all available
shared and case logs once, then analyzes all eligible cases together with all
submitted case and guest context. A separate eligible-ID list restricts diagnoses
without hiding context from cases whose evidence is unavailable. Transient
inference failures retry the whole suite with fresh agent/tool state. This avoids
mandatory agent-directed Drain calls for every log. The agent may still use
Drain and optional extractors when relevant. Snippet annotation is available
only to build agents; historical lookup can use pre-extracted suite evidence.
Its evidence references must
point to a pre-extracted snippet in the matching case or a shared log.
During result assembly, agent answers are matched by case ID; duplicate or
unknown IDs are rejected, while a missing or blank answer affects only its case.
The start prompt includes a compact inventory of successfully fetched log names
grouped by case and shared suite scope, without log URLs or contents. It directs
the agent to orient itself with shared run-level tmt evidence, then inspect a
failed test's main output for the cause, and to consider shared preparation or
pipeline logs for setup failures or missing case evidence. The inventory is
scoped to the current suite. When available,
`tmt-run.log` (published as `tmt-log` in some results) provides run-level stage,
outcome, and error clues; it does not replace a testcase's detailed output in
`testout.log`. The tmt workdir's `log.txt` provides deeper run context.
