# Agents

Build and Testing Farm agents have separate workflows. Shared extractor setup
and other utilities live in `logdetective.agents.common`.

Behavior of tools used by agents is described in [tools.md](./tools.md).

Agent runs are executed as tasks in [worker](./workers.md) pods.
Both workflows reject a final answer when the provider reports an output-token
limit, even if the partial answer can be parsed. This is a task error, not a
successful build diagnosis or a set of missing suite cases.

## Prompts

Jinja templates for system prompts and first messages sent to agents are placed in
location specified by `LOGDETECTIVE_PROMPT_TEMPLATES` (`logdetective/prompts` by default) environment variable.

It is possible to add extra sources/references for agents via `server/config.yml` file (`prompts` section, under `references`).

All system prompt templates must include place for `system_time` variable.

If `references` list is defined in `server/config.yml`, templates must also include a handling for a list of references.

Example:

```jinja
{% if references %}
## References:

    {% for reference in references %}
    * {{ reference.name }} : {{ reference.link }}
    {% endfor %}
{% endif %}

```

## Build analysis agent

The base agent offer analysis of failed package builds. The built in prompt
is written with RPM ecosystem in mind, and must be modified in production
if other types of processes are meant to be analyzed.

The analysis is tracked under `generic` task type, and metrics for execution
are available under `/metrics/analyze` route.

### Snippet extraction

Agent is constrained so that it has to extract snippets from all submitted
artifacts using the `drain_extractor` tool. The agent can then use all
available extractors to refine findings.

### Other tools

Agent uses `ThinkTool` provided by beeai-framework as a de-facto reasoning mechanism.
This has only a limited benefit Log Detective is deployed with a sufficiently
capable reasoning model. Usage of the `ThinkTool` is enforced during the first step of the analysis. At most `2` calls of the the `ThinkTool` are allowed.

Agent is also allowed to use other extractors, `annotated_snippet_lookup`
and `snippet_annotation` tool, if they are enabled in config.

### Agent output

The build agent returns an `AgentResponse` with a structured `Explanation` and
an optional structured `Solution`. The [public API result](api.md#completed-normal-analysis)
exposes their text as `explanation` and `solution`, not as nested objects.
The `no_issue_found` flag means the agent found no evidence of an issue; it
does not guarantee that the build was error-free.

## Testing Farm suite analysis agent

Specialized agent for analysis of Testing Farm results operates
in a test suite scope.
Agent receives artifacts for all contained test cases, as well as shared artifact
and information about test environment.
There is no cap on the number of submitted test cases. A suite may include up to
`max_shared_artifacts` shared artifacts. Each test case may include up to
`max_artifacts_per_case` artifacts. Together, all shared and case artifact entries
must fit within the suite-wide `max_total_artifacts` limit.

The task is tracked as `tft_suite`, with metrics available under `/metrics/analyze-tft` route.

Before the agent is executed, snippets from all shared and case artifacts are
extracted using `drain_extractor` tool at once, and snippets are included in
the first agent message.
This minimizes number of tool calls the agent has to make.
A transient inference retry reuses the completed extraction, even if it found no
snippets. Each attempt gets fresh agent state but keeps the original deadline.

Agent may still use extractor tools during analysis process.

### Prompt

The start prompt includes an inventory of successfully fetched artifacts by case and shared suite.
It directs the agent to use shared suite level tmt evidence first, then inspect
failed test's main output for the cause, and to consider shared preparation or
pipeline logs for setup failures or missing case evidence.

### Available tools

Testing Farm suite analysis agent can use all configured extractor tools.
The agent doesn't have access to the `snippet_annotation`, or `Thinking tool`,
as they would offer only a limited benefit.

### Agent output

The agent receives context for all submitted cases, but its `cases` list may
contain diagnoses only for eligible case IDs. Each diagnosis identifies the
case by `case_id` and gives an explanation, an optional solution, and the same
`no_issue_found` flag as the build agent.

The [public API result](api.md#completed-testing-farm-analysis) reports an
outcome for each submitted case, keyed by ID. A case with no usable evidence
gets `artifact_error`; an eligible case with no diagnosis or a blank explanation
gets `missing_analysis`. Neither error discards diagnoses for other cases.

Only when the request contains no cases should the agent return a suite-wide
diagnosis in `suite_fallback` and leave `cases` empty. This diagnosis needs a
nonblank explanation, but no case ID.

### Evidence

The agent is prompted to reference information used for analysis by file and line
of origin in the `explanation` field for each test case.

To improve actionability, and simplify verification of the the analysis,
an index is constructed using all snippets extracted.

The internal index groups variants by filename and starting line. Identical text
at the same location is deduplicated; distinct variants are kept. In the API
response, shared snippets are grouped by filename, and each snippet retains its
own `line_number`.

The index of evidence is used together with agent output to construct the API response.
