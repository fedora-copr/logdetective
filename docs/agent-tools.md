# Agent tools

Build and Testing Farm agents have separate workflows. Shared extractor setup
and result utilities live in `logdetective.agents.common`; the suite workflow
pre-extracts shared and case evidence before one whole-suite agent run.
The shared setup registers each tool with its call requirements and tracks
extractors in the same order used for snippet annotation and final responses.

## Build analysis

Build requests supply `files` and optional `build_metadata`; Testing Farm
requests use a separate suite-specific context and log structure.

### Required and optional extraction

The agent must call `drain_extractor` once for every build artifact before
using optional extractors or providing the final answer. Other extractors do
not satisfy this requirement. When available, `csgrep_extractor` captures GCC
compiler traces and `python_traceback_extractor` captures Python exception
tracebacks. The start prompt mentions each optional tool only when the current
agent run offers it. BeeAI's Drain invocation requirement remains the hard
gate; prompt ordering is guidance to reduce invalid attempts.

### Historical lookup

When the annotation database contains examples and an embedding model is
available, `annotated_snippet_lookup` can search for similar failures. The agent
passes a specific extracted error as `snippet`. Returned annotations, build
problems, and solutions describe past builds.

### Snippet annotation

`snippet_annotation` attaches a brief description to a useful extracted snippet.
Its `source_file` and `line_number` must match extractor output. The description
goes in the `snippet_analysis` argument.

## Testing Farm suite analysis

### Suite execution and retries

For `tft_suite` tasks, the worker runs Drain extraction for all available
shared and case artifacts once, then analyzes all eligible cases together with all
submitted case and guest context. A separate eligible-ID list restricts diagnoses
without hiding context from cases whose evidence is unavailable. Transient
inference failures retry the whole suite with fresh agent/tool state. This avoids
mandatory agent-directed Drain calls for every log. The agent may still use
Drain and optional extractors when relevant. Snippet annotation is available
only to build agents; historical lookup can use pre-extracted suite evidence.

The suite agent validates returned case IDs once, rejecting duplicate or
ineligible IDs for the entire suite. Result assembly consumes those validated
answers without repeating the ID checks; missing or empty explanations remain
case-local `missing_analysis` outcomes.

### Prose citations and attached evidence

The agent is asked to support its explanation with prose citations using supplied
filenames and starting line numbers from its own case or shared artifacts. These are
model-written claims: result assembly neither parses nor validates them, and an
incorrect citation does not discard the explanation. No structured citation list
is requested; unsolicited citation fields are ignored.

The application independently attaches own-case snippets to each successful
case. Shared snippets appear once in the suite-level `shared_snippets` index,
keyed by `source_file`, with each snippet retaining its own `line_number`.
Cases and suite fallback carry `shared_snippet_refs` to those keys; fallback
has no inline snippets. Sibling-case snippets are never attached. References
are application-generated, not model citations. Locations retain collection order,
with distinct variants at each location in pre-extraction then extractor order.
This preserves richer optional evidence, such as complete tracebacks overlapping
Drain fragments. Identical variants are deduplicated.
Each successful attempt assembles its evidence independently;
failed attempts do not modify the pre-extracted retry baseline.

During result assembly, agent answers are matched by case ID; duplicate or
unknown IDs are rejected, while a missing or blank answer affects only its case.

### Prompts and log selection

The concise system prompt covers lifecycle, evidence grounding, untrusted-data
handling, and conditional remediation. The suite-start prompt consolidates case
eligibility, citation/attachment rules, and log-selection guidance. This wording
does not reduce the supplied context or extracted evidence, change the structured
response contract, or override BeeAI's framework templates.

The start prompt includes a compact inventory of successfully fetched log names
grouped by case and shared suite scope, without log URLs or contents. It directs
the agent to orient itself with shared run-level tmt evidence, then inspect a
failed test's main output for the cause, and to consider shared preparation or
pipeline logs for setup failures or missing case evidence. The inventory is
scoped to the current suite. When available,
`tmt-run.log` (published as `tmt-log` in some results) provides run-level stage,
outcome, and error clues; it does not replace a testcase's detailed output in
`testout.log`. The tmt workdir's `log.txt` provides deeper run context.

## Cancellation and timeouts

All analysis tasks (generic, Koji, GitLab, and Testing Farm) share one positive,
integer `general.analysis_timeout` (default 1200 seconds). The worker establishes
an absolute event-loop deadline after successful `mark_started()`, immediately
before execution. Retrieval, sanitization, setup, extraction, inference, retry
backoff, and result assembly consume it. Every retry receives fresh agent and
controller state but the same deadline; completed suite baseline evidence is reused.
Phase-boundary checks reject synchronous overruns, including empty-artifact paths.
Expiry produces a task-level `timeout`, not partial results.

Queue waiting, terminal state publication, and necessary cleanup are excluded.
GitLab preparation returns a typed in-memory diagnosis/context payload; only after
successful deadline validation does untimed publication suppress old comments,
render and post/update comments (including the five-second delay), and persist
records and metrics. Clients remain open through publication and close before
terminal state publication. Explicit cancellation and existing HTTP/provider/Koji
limits still apply. No automatic job retries or queue middleware are added.

Before admission and again before retrieval, `testing_farm.max_artifacts_per_case`,
`max_shared_artifacts`, and `max_total_artifacts` bound submitted entries (defaults 10, 10,
and 120). Repeated URLs count, although retrieval still caches them. These are
configurable positive integers, not fixed ceilings; there is no testcase cap.
API and worker deployments should use matching configuration.

Agent runs translate task cancellation into BeeAI abort signals and join the
run before returning. A task-local blocking-work scope follows nested tool
contexts, waits for their already-started threads, and prevents detached tool
contexts from starting new blocking work after the run closes. This applies
to both timeout and explicit cancellation, including repeated cancellation.

### Cancellation reporting and cleanup

`run_blocking()` joins shielded workers through repeated cancellation without
clearing the caller's cancellation counts. The enclosing timeout removes only its
own request: timeout alone reports `timeout`, while overlapping explicit
cancellation remains cancellation and is acknowledged immediately after cleanup.
Database publication fences remain authoritative.

Python cannot forcibly stop a running thread. Cleanup waits for it to return and
can hold resources indefinitely if it never does; the execution deadline is not
a hard cleanup bound. Executor threads may be reused after the call finishes.
