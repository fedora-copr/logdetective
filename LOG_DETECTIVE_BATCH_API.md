# Testing Farm per-suite API

Status: server contract implemented in `/analyze/tft`; gluetool integration is deferred.

One POST represents one tmt schedule entry/suite, regardless of guest count. The body contains `suite` context, a `guests` map, `shared_logs`, and `cases`. Every eligible failed, errored, or `needs_inspection` testcase is included with a unique request-local `case_id`. Zero cases require a suite error state/result or error details. Only safe serialized environment, testcase, check/subresult, and published log fields are accepted. Arbitrary properties, secrets, local paths, and inline logs are excluded.

There is no testcase-count or log-reference limit; the existing 50 MiB aggregate artifact budget applies. Over-limit requests are rejected, not chunked or silently reduced. URL and SSRF controls match the generic endpoint.

Admission returns the existing 202 task envelope with `taskType: "tft_suite"`, `deadlineAt`, `Location`, and `Retry-After`. GET/DELETE `/tasks/{id}` are unchanged. A done result maps `cases[case_id]` to an analysis or safe per-case error. With no cases, `suite_fallback` holds one analysis. Results and retrieval warnings are published only after completion.

The worker caches URL successes and failures, preserves partial evidence, namespaces artifacts by case, pre-extracts Drain evidence once, and invokes one Testing Farm agent for the suite. Optional extraction tools and conditional HTTP-429 advice remain. One agent run may make multiple provider requests. A case with no usable own or shared log, or missing/empty case output, produces a per-case error; duplicate/unknown IDs or invalid evidence references fail validation. Global inference failures, timeouts, and cancellation remain task-level outcomes. The admission-to-publication suite budget defaults to 600 seconds; reconciliation and blocking cleanup can make terminal reporting later.

After the API is finalized and verified, gluetool-modules can submit at its existing per-suite serialization hook, map results by `case_id`, preserve cached results, and use a 600-second report-wait budget. No gluetool files are changed in this phase.
