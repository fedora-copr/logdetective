# Threat Model: Log Detective

## 1. System context

Log Detective is a tool and web service that analyzes RPM build failure logs
using large language models. It serves Fedora, CentOS Stream, and RHEL
packagers by extracting representative snippets from build logs (via the Drain3
template-mining algorithm, csgrep, and Python traceback extraction) and sending
them to a language model for failure explanation and solution suggestions.

The system has a **FastAPI API tier** and a separate **Procrastinate worker
tier**. It exposes an asynchronous REST API for log analysis, integrates with
GitLab (webhook-driven merge request comments on failed builds), and retrieves
RPM build tasks from Koji. Durable application state, Procrastinate jobs, and
operational metrics are stored in PostgreSQL. Metrics include request counts and
average response time, analysis completion time, and response length, grouped
by hour, day, or month.

The server is deployed either on an AWS VM using podman-compose
or on Red Hat OpenShift.

The inference backend can be a local vLLM or llama.cpp instance,
or any external service that implements the OpenAI API. The production
podman-compose configuration deploys multiple local vLLM GPU instances behind
nginx; external cloud-based inference is also supported.

**Security assumptions:** The server runs behind TLS — via Gunicorn with
cert/key in the podman-compose deployment, or via OpenShift Routes (edge or
re-encrypt TLS termination) in the OpenShift deployment. In the podman-compose
deployment, inference backends are on a private container network (using
`expose:`, not published to the host); PostgreSQL uses `ports:` mapping,
publishing the database port to the host — access depends on host-level
firewall configuration.

In the OpenShift deployment, network isolation is provided by NetworkPolicies and namespace
boundaries; credentials are stored in OpenShift Secrets rather than on-disk
config files. When using external cloud-based inference, inference API keys and
all log snippets sent for analysis transit the public internet over TLS.

The shared application configuration, separate API token file, environment
file, and OpenShift Secrets are operator-managed with appropriate access
controls. The API and worker deliberately share application configuration,
including inference, outbound GitLab, webhook, and Koji credentials. API bearer
tokens remain in a separate server-specific file.

Authentication is optional — if `LOGDETECTIVE_TOKENS_FILE` is unset, all endpoints are open, which is intended for
development only. SSRF protection is applied to user-supplied log URLs but not
uniformly to all outbound HTTP requests. The system processes untrusted log
content but sanitizes personal identifiers (emails, GPG fingerprints, RSA keys,
and public key identifiers) before forwarding to the LLM.


## 2. Assets

| asset | description | sensitivity |
|---|---|---|
| api_auth_tokens | Named bearer tokens loaded from `LOGDETECTIVE_TOKENS_FILE` and controlling access to all API endpoints; only names are stored with metrics | critical |
| gitlab_api_tokens | Per-instance values in the shared application configuration granting read/write access to GitLab projects and merge requests | critical |
| inference_api_credentials | API keys and credentials for LLM inference backends (OpenAI, Vertex AI, Gemini, and any OpenAI-compatible backend such as vLLM or llama.cpp) | critical |
| application_config_file | Shared YAML config containing operational parameters, inference credentials, outbound GitLab API tokens, inbound GitLab webhook secrets, and Koji caller secrets | critical |
| gitlab_webhook_secrets | Per-instance secrets validating inbound webhook requests | high |
| koji_auth_tokens | Per-instance server-side access tokens that callers must present in the `X-Koji-Token` header to use logdetective's Koji analysis endpoints; these are not credentials for authenticating to Koji itself (the Koji XMLRPC connection is unauthenticated) | high |
| postgresql_credentials | Database username and password for metrics, comments, and analysis storage | high |
| openshift_secrets | OpenShift Secret objects storing API tokens, inference credentials, and database passwords; accessible via Kubernetes API to principals with namespace-level RBAC | critical |
| openshift_route_config | OpenShift Route definitions controlling TLS termination policy and external access to the service | high |
| build_log_content | RPM build logs submitted by users or fetched from URLs; may contain paths, hostnames, package names, email addresses, GPG fingerprints, RSA keys, and public key identifiers (the latter are sanitized before LLM submission but present in raw logs) | medium |
| llm_analysis_responses | AI-generated explanations posted as GitLab MR comments and stored in database | medium |
| service_availability | Ability for legitimate users to submit and complete analysis; depends on PostgreSQL, the Procrastinate worker pool, and inference capacity | high |
| metrics_data | Aggregated request volume, response and analysis timing, response length, endpoint usage, and named-token attribution stored in PostgreSQL and exposed through the metrics API | medium |
| database_records | Durable task inputs and results, Procrastinate jobs, metrics, MR job records, and annotated builds | medium |

## 3. Entry points & trust boundaries

| entry_point | description | trust_boundary | reachable_assets |
|---|---|---|---|
| POST /analyze | Validates input and atomically admits generic analysis; returns an opaque task UUID and no result | remote unauth/auth peer → API process → PostgreSQL | build_log_content, database_records |
| POST /analyze/rpmbuild/koji | Atomically admits Koji analysis from JSON `kojiInstance` and `taskId`; `X-Koji-Token` is required only when the instance token list is non-empty | remote unauth/auth peer → API process → PostgreSQL | database_records, koji_auth_tokens |
| GET, DELETE /tasks/{task_id} | Reads or cancels the caller-owned operation using an opaque UUID; does not expose the internal Procrastinate job ID | remote unauth/auth peer → API process → PostgreSQL | llm_analysis_responses, database_records |
| POST /webhook/gitlab/job_events | Validates and durably admits a GitLab webhook before returning 204; the worker later downloads logs and posts comments | remote unauth/auth peer → API process → PostgreSQL → worker | build_log_content, llm_analysis_responses, database_records, gitlab_api_tokens |
| GET /metrics/{route}/ | Metrics endpoint returning request count, average response time, average analysis completion time, and average response length in a columnar response. Callers supply `start_time`, optional `end_time`, `time_period` (`hour`, `day`, or `month`), and an optional non-secret `api_token_name` filter. The endpoint is subject to global bearer authentication like all other routes, but filtering is not scoped to the authenticated token name | remote unauth/auth peer → application process | metrics_data, database_records |
| GET /version | Returns application version string; subject to global bearer auth when a token file is configured, unauthenticated when unset | remote unauth/auth peer → application process | — |
| user-supplied log URLs | URLs in `/analyze` payload fetched by a worker; pass through the SSRF-protected resolver | worker process → remote server (SSRF boundary) | build_log_content |
| analysis worker task | Procrastinate runs analysis directly as async work; blocking client and CPU calls are tracked until completion | Procrastinate worker → inference and integration clients | build_log_content, inference_api_credentials |
| openshift_api | Kubernetes/OpenShift API server reachable from within the pod (kubernetes.default.svc); accepts service account tokens auto-mounted into pods | application process → cluster control plane (pod → API server boundary) | openshift_secrets, openshift_route_config |
| external_inference_endpoint | Outbound requests from workers carrying API keys and sanitized log snippets | worker process → external inference service (TLS boundary) | inference_api_credentials, build_log_content |
| environment_and_config | Shared application config, API token file, DB credentials, or OpenShift Secrets loaded by application processes | local admin / OpenShift RBAC → API or worker process | api_auth_tokens, postgresql_credentials, inference_api_credentials, gitlab_api_tokens, openshift_secrets |

## 4. Threats

| id | threat | actor | surface | asset | impact | likelihood | status | controls | evidence |
|---|---|---|---|---|---|---|---|---|---|
| T15 | Overly permissive OpenShift RBAC allows unauthorized namespace users to read Secret objects containing inference API keys, database credentials, and auth tokens via the Kubernetes API | local_user | openshift_api | openshift_secrets, inference_api_credentials, api_auth_tokens, postgresql_credentials | critical | rare | partially_mitigated | OpenShift RBAC restricts Secret access by default to namespace admins; service account tokens are auto-mounted unless explicitly disabled | Default service account may have broader permissions than needed; `get secrets` permission grants access to all secrets in the namespace |
| T5 | Unauthenticated attacker floods `/analyze` with large payloads or many operations, growing the durable queue and exhausting database, worker, or inference capacity | remote_unauth | POST /analyze | service_availability | high | likely | partially_mitigated | Content-Length and file-count limits; bounded worker concurrency; durable admission keeps API workers available; optional bearer authentication | No rate limit or admission-depth limit; accepted work consumes database space and inference capacity |
| T6 | Attacker submits crafted log content designed to manipulate LLM behavior (prompt injection), causing misleading analysis, fabricated solutions, or attacker-chosen text in explanations posted as GitLab MR comments | remote_unauth | POST /analyze, POST /webhook/gitlab/job_events | llm_analysis_responses | high | possible | partially_mitigated | Input sanitization (PII redacted); Drain3/csgrep/traceback extractors reduce attacker-controlled text reaching the LLM; chat-API role separation places log content in user message, instructions in system message; server path enforces `AgentResponse` schema validation on LLM output (agent.py:172,188); agent tools are sandboxed to extraction logic only; MR comments carry AI disclaimer and abuse-reporting link | Attacker-controlled log content is embedded verbatim in the LLM prompt after snippet extraction; no sentinel boundaries within user messages; no prompt-injection-specific input filtering or secondary classifier |
| T7 | Global bearer-token authentication is disabled when `LOGDETECTIVE_TOKENS_FILE` is unset; misconfigured production deployment exposes `/analyze`, `/version`, and `/metrics` without auth; Koji and GitLab webhook endpoints retain their independent header checks | local_admin | POST /analyze, GET /version, GET /metrics | llm_analysis_responses, metrics_data, database_records | high | possible | partially_mitigated | Documented behavior; invalid or missing configured token files fail startup; named token values are compared in constant time and bearer-token values are absent from errors and metrics | An operator can still omit both authentication environment variables |
| T16 | Compromised application pod uses auto-mounted service account token to access the Kubernetes API, escalating from application-level compromise to cluster-level reconnaissance, lateral movement, or authenticated Secret retrieval; this is the only path from application compromise to credential exfiltration via the Kubernetes API, since unauthenticated requests are rejected | remote_auth | openshift_api | openshift_secrets, openshift_route_config, inference_api_credentials | high | rare | partially_mitigated | OpenShift restricts default service account permissions; SecurityContextConstraints (SCCs) limit pod capabilities | Service account token auto-mounted at /var/run/secrets/kubernetes.io/serviceaccount unless `automountServiceAccountToken: false` is set; Kubernetes API is reachable from any pod at kubernetes.default.svc; if the service account has `get secrets` permission, this token provides authenticated access to all Secrets in the namespace |
| T17 | A caller queries metrics for another client's known or guessed non-secret token name, learning that client's request volume and aggregate timing or response-size patterns | remote_auth / remote_unauth | GET /metrics/{route}/ | metrics_data | medium | possible | partially_mitigated | Metrics contain aggregates rather than raw requests, logs, responses, or bearer-token values; global bearer authentication applies when configured; invalid route, timestamp, and period values are rejected by FastAPI/Pydantic | The `api_token_name` query parameter is passed directly to the aggregate filter and is not constrained to `request.state.api_token_name`; when global authentication is disabled, any remote caller can make the same queries (`server.py:308-335`) |
| T18 | A caller repeatedly requests a very large date range at hourly granularity, forcing PostgreSQL to scan, group, and return a potentially large number of metric buckets and consuming database, application, and network capacity | remote_auth / remote_unauth | GET /metrics/{route}/ | service_availability | medium | possible | partially_mitigated | Aggregation occurs in PostgreSQL; request timestamps and endpoints are indexed; `time_period` is restricted to hour, day, or month; global bearer authentication applies when configured | No maximum query duration, bucket count, pagination, response-size limit, rate limit, or per-client concurrency limit is enforced; `start_time` and `end_time` are caller-controlled (`server.py:308-335`, `database/models/metrics.py:184-240`) |
| T9 | A compromised or lost analysis worker leaves expensive work ambiguous or publishes a stale result after cancellation | remote_unauth / infrastructure failure | worker and task cancellation | service_availability, database_records | medium | possible | mitigated | No automatic analysis retry; dedicated maintenance worker reserves reconciliation capacity; application generation fences terminal publication; stalled jobs and missing or unexpectedly terminal jobs outside cancellation confirmation fail their application tasks; succeeded jobs with non-cancelling active tasks and no application outcome are also failed; cancellation is durable before best-effort abort; cooperative cancellation stops async work and waits for tracked blocking calls before acknowledgement; network clients have bounded timeouts | A hard host failure can delay reconciliation; already-sent external requests or side effects cannot be recalled |
| T14 | Build log content (containing hostnames, internal paths, build environment details, and package metadata) is transmitted to an external cloud inference provider; compromise or misconfiguration of the provider, or overly broad data retention policies, exposes internal infrastructure details to a third party | supply_chain | external_inference_endpoint | build_log_content | medium | possible | partially_mitigated | PII sanitization removes emails, GPG fingerprints, RSA keys, and public key identifiers before sending; Drain3 extracts only representative snippets, reducing volume; TLS in transit | No controls on what the inference provider retains or how it processes the data beyond the provider's own terms of service; full snippet content is sent including hostnames and paths |
| T11 | Attacker sends oversized or specially crafted zip archives via GitLab artifacts, causing excessive memory consumption or zip bomb decompression | remote_unauth / remote_auth | POST /webhook/gitlab/job_events | service_availability | medium | rare | partially_mitigated | Content-Length check on artifacts zip; `zipfile.open()` per entry (no `extractall()`); max_artifact_size configurable | No zip compression ratio check; on-disk TemporaryFile for zip download, individual entries decompressed into memory via `zipfile.open().read()` |
| T13 | When a Koji instance token list is empty, an unauthenticated caller can submit analyses for arbitrary public Koji task IDs and consume capacity | remote_unauth / remote_auth | POST /analyze/rpmbuild/koji | service_availability | low | possible | partially_mitigated | Results use opaque UUIDv4 task resources and cannot be enumerated by Koji ID; bounded worker concurrency | Empty token lists intentionally bypass Koji-specific authentication; global bearer authentication may also be disabled |
| T19 | Caller guesses or obtains another client's task UUID and attempts to read or cancel it | remote_auth / remote_unauth | GET, DELETE /tasks/{task_id} | llm_analysis_responses, database_records | medium | rare | mitigated | UUIDv4 identifiers; configured bearer-token name is stored as owner and mismatches return 404; cancellation update targets one task and result publication is fenced | In unauthenticated development mode all tasks share the `NULL` owner, so UUID secrecy is the only authorization barrier |
| T20 | Compromise of any application container exposes credentials used by the other application tier because API and worker share one configuration trust domain | remote_auth / remote_unauth | environment_and_config | inference_api_credentials, gitlab_api_tokens, gitlab_webhook_secrets, koji_auth_tokens | high | possible | accepted | Shared configuration is an explicit operational choice; config files should have restrictive host permissions and must not be logged or committed with production values | An API compromise can expose inference and outbound GitLab credentials even though the API does not use them directly; a worker compromise can expose inbound webhook and Koji credentials |

## 5. Deprioritized

| threat | reason |
|---|---|
| Physical access to server hardware | Out of scope; server is deployed on managed infrastructure (AWS or OpenShift cluster) with standard physical security controls |
| SQL injection through application database interactions (formerly T12) | No exploitable path exists in the current code. Database operations use SQLAlchemy expressions and parameterized values; metrics aggregation granularity is constrained to the `TimePeriod` enum; and the metrics query's `literal_column` values are static internal labels rather than request values (`database/models/metrics.py:183-214`). Retain this as a regression consideration if raw SQL or dynamic SQL identifiers are introduced. A regression could expose or modify `database_records` and `metrics_data`; PostgreSQL credentials are supplied through the environment and are not stored in the application database. |
| Direct attacks on local inference backend (vLLM/llama.cpp) | Applies only to podman-compose deployments with local inference; backends are on a private container network, not exposed to the internet. When using external cloud inference, the provider's security posture is covered by T14 |
| PostgreSQL network attacks | In podman-compose deployments, PostgreSQL port is published to the host via `ports:` mapping (all compose variants); access depends on host firewall and network configuration. In OpenShift, database is behind NetworkPolicies within the namespace. Risk is deployment-specific and mitigated by host-level network controls |
| DNS rebinding against SSRF protection | SSRFProtectedResolver checks resolved IPs at connect time, mitigating basic DNS rebinding; advanced TOCTOU attacks require precise timing and are impractical against this service |
| Malicious prompt templates on disk | Prompt templates are loaded from operator-managed files; an attacker with write access to these files already has sufficient access to compromise the system directly |
| CORS-based cross-origin attacks | No CORSMiddleware is configured; without it, FastAPI returns responses without `Access-Control-Allow-Origin` headers, so browsers block client-side JavaScript from reading responses (Same-Origin Policy). The server still processes and returns the request — the browser enforces the restriction, not the server. For this JSON API (`Content-Type: application/json`), browsers send a preflight OPTIONS request first, and without CORS headers in the preflight response, the actual request is not sent. This is the secure default for an API-only service |
| Supply chain compromise of generic Python dependencies | Generic risk applicable to any Python project. General controls: poetry.lock version pinning, hash-based dependency verification |

## 6. Open questions

- Are the GitLab API tokens scoped with minimum necessary permissions (e.g., read-only for project/job data, write-only for MR comments)?
- What is the intended behavior when `webhook_secrets` is empty for a GitLab instance in production? The current code treats all requests as authorized.
- Is there a process for verifying integrity of models pulled from HuggingFace before deployment?
- The `packages` code default is an empty list (deny all), but the shipped config file sets `packages: [".*"]` (all packages eligible for MR comments) — is this intentional for production, or should it be restricted to specific package namespaces?
- Is `automountServiceAccountToken: false` set on the application pod spec in OpenShift, or does the pod carry an unnecessary service account token?
- Are OpenShift NetworkPolicies configured to restrict pod egress to only the required destinations (inference provider, GitLab, Koji, PostgreSQL), or is egress unrestricted?
- What is the data retention and processing policy of the external cloud inference provider? Does it comply with the data handling requirements for the build log content being analyzed?
- Should metrics access be restricted to operators, or should authenticated callers be limited to metrics attributed to their own token name?
- What maximum metrics date range or bucket count can production PostgreSQL and application workers serve safely?

## 7. Provenance

- mode: bootstrap-then-interview
- date: 2026-09-18
- target: https://github.com/fedora-copr/logdetective @ 6340a5a6
- inputs: source code review, pyproject.toml dependency analysis, git commit history (security-related commits), docker-compose and deployment configuration, OpenShift deployment context (Routes, Secrets, NetworkPolicies, RBAC), inference service integration, THREAT_MODEL_README.md format specification
- owner: unset

## 8. Recommended mitigations

| mitigation | threat_ids | closes_class | effort |
|---|---|---|---|
| Add per-IP or per-token rate limiting middleware (e.g., slowapi) to `/analyze` and webhook endpoints | T5, T9 | partial | M |
| Treat log text as untrusted data: strengthen delimitation of user-supplied content from system instructions beyond chat-API role separation (e.g., XML-tag wrapping or sentinel boundaries within user messages); apply content-level filtering or a secondary classifier as defense in depth — note that tool use cannot be disabled on the server path since extraction tools are architecturally essential, but tools are sandboxed to extraction logic only | T6 | partial | M |
| Add per-client admission quotas or a maximum durable queue depth to prevent unbounded accepted work | T5 | partial | M |
| Set `automountServiceAccountToken: false` on the application pod spec to prevent unnecessary Kubernetes API access from a compromised pod | T16 | yes | S |
| Apply least-privilege RBAC: ensure the application's service account has no `get`/`list` permission on Secrets; restrict namespace access to operators only | T15 | yes | S |
| Establish a data processing agreement with the external inference provider covering retention limits, access controls, and breach notification for build log content | T14 | yes | M |
| Add zip decompression ratio check and per-entry size limit to prevent zip bomb attacks via GitLab artifacts; consider streaming decompression instead of reading entire entries into memory | T11 | partial | S |
| Require non-empty Koji token list for production deployments; add a startup warning or validation error when `tokens` is empty in non-development environments | T13 | yes | S |
| Configure egress NetworkPolicies in OpenShift to restrict outbound traffic from the application pod to only required destinations (inference provider, GitLab, Koji, PostgreSQL) | T16 | partial | M |
| Authorize metrics by role or bind `api_token_name` filtering to the authenticated token identity; reserve cross-token aggregate access for operators | T17 | yes | M |
| Enforce a maximum metrics range or bucket count, reject `start_time >= end_time`, and select a coarser minimum granularity for long ranges | T18 | partial | S |
| Apply per-token/IP rate and concurrency limits to metrics queries in addition to analysis endpoints | T18 | partial | M |
