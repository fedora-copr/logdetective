# Threat Model: Log Detective

## 1. System context

Log Detective is a tool and web service that analyzes RPM build failure logs
using large language models. It serves Fedora, CentOS Stream, and RHEL
packagers by extracting representative snippets from build logs (via the Drain3
template-mining algorithm, csgrep, and Python traceback extraction) and sending
them to a language model for failure explanation and solution suggestions.

The system operates in one of two mutually exclusive modes — never both at the
same time. As a **CLI tool** (`logdetective` command), it loads a local GGUF
model via llama-cpp-python, fetches logs from a URL or local path, extracts
snippets, and prints an LLM-generated explanation. As a **FastAPI server**
it exposes a REST API for log analysis, integrates with GitLab (webhook-driven merge
request comments on failed builds) and Koji (RPM build task analysis with callback notifications),
storing metrics in PostgreSQL.

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

The configuration file, env file or OpenShift Secrets (containing API tokens,
webhook secrets, and inference credentials) are operator-managed with appropriate access controls.

Authentication is optional — if `LOGDETECTIVE_TOKEN` is unset, all endpoints are open, which is intended for
development only. SSRF protection is applied to user-supplied log URLs but not
uniformly to all outbound HTTP requests. The system processes untrusted log
content but sanitizes personal identifiers (emails, GPG fingerprints, RSA keys,
and public key identifiers) before forwarding to the LLM.


## 2. Assets

| asset | description | sensitivity |
|---|---|---|
| api_auth_token | Bearer token (`LOGDETECTIVE_TOKEN`) controlling access to all API endpoints | critical |
| gitlab_api_tokens | Per-instance `api_token` values granting read/write access to GitLab projects and merge requests | critical |
| inference_api_credentials | API keys and credentials for LLM inference backends (OpenAI, Vertex AI, Gemini, and any OpenAI-compatible backend such as vLLM or llama.cpp); in CLI mode, llama-cpp-python loads GGUF models natively without API keys | critical |
| server_config_file | YAML config containing some credentials, endpoint URLs, and operational parameters | critical |
| gitlab_webhook_secrets | Per-instance secrets validating inbound webhook requests | high |
| koji_auth_tokens | Per-instance server-side access tokens that callers must present in the `X-Koji-Token` header to use logdetective's Koji analysis endpoints; these are not credentials for authenticating to Koji itself (the Koji XMLRPC connection is unauthenticated) | high |
| postgresql_credentials | Database username and password for metrics, comments, and analysis storage | high |
| openshift_secrets | OpenShift Secret objects storing API tokens, inference credentials, and database passwords; accessible via Kubernetes API to principals with namespace-level RBAC | critical |
| openshift_route_config | OpenShift Route definitions controlling TLS termination policy and external access to the service | high |
| build_log_content | RPM build logs submitted by users or fetched from URLs; may contain paths, hostnames, package names, email addresses, GPG fingerprints, RSA keys, and public key identifiers (the latter are sanitized before LLM submission but present in raw logs) | medium |
| llm_analysis_responses | AI-generated explanations posted as GitLab MR comments and stored in database | medium |
| service_availability | Ability for legitimate users to submit logs and receive analysis results; depends on Gunicorn worker pool and inference backend capacity | high |
| database_records | Metrics, MR job records, Koji task analyses, annotated builds, emoji feedback | medium |

## 3. Entry points & trust boundaries

| entry_point | description | trust_boundary | reachable_assets |
|---|---|---|---|
| POST /analyze | Main analysis endpoint; accepts JSON with file list (raw content or URLs) and optional build metadata | remote unauth/auth peer → application process | build_log_content, llm_analysis_responses, database_records |
| GET /analyze/rpmbuild/koji/{instance}/{task_id} | Retrieve existing Koji task analysis result | remote unauth/auth peer (X-Koji-Token required only when instance `tokens` list is non-empty) → application process | llm_analysis_responses, database_records |
| POST /analyze/rpmbuild/koji/{instance}/{task_id} | Trigger Koji task analysis; accepts callback URL via X-Koji-Callback header | remote unauth/auth peer (X-Koji-Token required only when instance `tokens` list is non-empty) → application process | build_log_content, llm_analysis_responses, database_records, koji_auth_tokens |
| POST /webhook/gitlab/job_events | GitLab webhook for failed CI/CD jobs; triggers log download, analysis, and MR comment posting; requires `X-Gitlab-Instance` header to identify the source instance | remote unauth/auth peer (X-Gitlab-Token required only when instance `webhook_secrets` is non-empty; X-Gitlab-Instance always required) → application process | build_log_content, llm_analysis_responses, database_records, gitlab_api_tokens |
| POST /webhook/gitlab/emoji_events | GitLab webhook for emoji reactions on MR comments; requires `X-Gitlab-Instance` header | remote unauth/auth peer (X-Gitlab-Token required only when instance `webhook_secrets` is non-empty; X-Gitlab-Instance always required) → application process | database_records |
| GET /metrics/{route}/{metric_type} | Metrics endpoint returning request/response/emoji statistics; subject to global `LOGDETECTIVE_TOKEN` auth like all other routes | remote unauth/auth peer → application process | database_records |
| GET /version | Returns application version string; subject to global `LOGDETECTIVE_TOKEN` auth when set, unauthenticated when unset | remote unauth/auth peer → application process | — |
| CLI arguments | `file` (URL or local path), `--model` (HuggingFace name or local path), config file paths, prompt paths | local user → CLI process | build_log_content, llm_analysis_responses |
| user-supplied log URLs | URLs in /analyze payload fetched by server; pass through SSRF-protected resolver | application process → remote server (SSRF boundary) | build_log_content |
| koji_callback_url | X-Koji-Callback header value; server POSTs `{"task_id": N}` to this URL upon completion (no credentials, logs, or analysis data in payload); pass through SSRF-protected resolver | application process → external server | — |
| openshift_api | Kubernetes/OpenShift API server reachable from within the pod (kubernetes.default.svc); accepts service account tokens auto-mounted into pods | application process → cluster control plane (pod → API server boundary) | openshift_secrets, openshift_route_config |
| external_inference_endpoint | Outbound HTTPS requests to the cloud inference provider carrying API keys and log snippets | application process → external inference service (TLS boundary) | inference_api_credentials, build_log_content |
| environment_and_config | LOGDETECTIVE_TOKEN, LOGDETECTIVE_SERVER_CONF, DB credentials, config.yml or OpenShift Secrets loaded at startup | local admin / OpenShift RBAC → application process | api_auth_token, postgresql_credentials, inference_api_credentials, server_config_file, openshift_secrets |

## 4. Threats

| id | threat | actor | surface | asset | impact | likelihood | status | controls | evidence |
|---|---|---|---|---|---|---|---|---|---|
| T2 | Compromised or malicious GGUF model downloaded from HuggingFace without integrity verification leads to arbitrary code execution or backdoored inference | supply_chain | CLI arguments (--model) | llm_analysis_responses | critical | rare | unmitigated | none | `Llama.from_pretrained()` downloads without hash or signature verification |
| T15 | Overly permissive OpenShift RBAC allows unauthorized namespace users to read Secret objects containing inference API keys, database credentials, and auth tokens via the Kubernetes API | local_user | openshift_api | openshift_secrets, inference_api_credentials, api_auth_token, postgresql_credentials | critical | rare | partially_mitigated | OpenShift RBAC restricts Secret access by default to namespace admins; service account tokens are auto-mounted unless explicitly disabled | Default service account may have broader permissions than needed; `get secrets` permission grants access to all secrets in the namespace |
| T5 | Unauthenticated attacker floods /analyze with large payloads or many concurrent requests, exhausting inference capacity and blocking legitimate users | remote_unauth | POST /analyze | service_availability | high | likely | partially_mitigated | Content-Length limit (50 MiB), max 15 files per request, agent timeout (600s), Gunicorn worker count (16) | No rate limiting on any endpoint; 600s timeout ties up workers |
| T6 | Attacker submits crafted log content designed to manipulate LLM behavior (prompt injection), causing misleading analysis, fabricated solutions, or attacker-chosen text in explanations posted as GitLab MR comments | remote_unauth | POST /analyze, POST /webhook/gitlab/job_events | llm_analysis_responses | high | possible | partially_mitigated | Input sanitization (PII redacted); Drain3/csgrep/traceback extractors reduce attacker-controlled text reaching the LLM; chat-API role separation places log content in user message, instructions in system message; server path enforces `AgentResponse` schema validation on LLM output (agent.py:172,188); agent tools are sandboxed to extraction logic only; MR comments carry AI disclaimer and abuse-reporting link | Attacker-controlled log content is embedded verbatim in the LLM prompt after snippet extraction; no sentinel boundaries within user messages; CLI path has no output validation; no prompt-injection-specific input filtering or secondary classifier |
| T7 | Global bearer-token authentication (LOGDETECTIVE_TOKEN) disabled by default when env var is unset; misconfigured production deployment exposes `/analyze`, `/version`, and `/metrics` without auth; Koji and GitLab webhook endpoints retain their own independent header-based token checks (`X-Koji-Token`, `X-Gitlab-Token`) even when the global token is unset | local_admin | POST /analyze, GET /version, GET /metrics | llm_analysis_responses, database_records | high | possible | partially_mitigated | Documented behavior; separate dev/prod compose files; Koji and GitLab endpoints enforce per-instance token validation independently | `requires_token_when_set()` silently passes when env var unset (server.py:202); app-level `Depends(requires_token_when_set)` applied to all routes including /metrics (server.py:239) |
| T16 | Compromised application pod uses auto-mounted service account token to access the Kubernetes API, escalating from application-level compromise to cluster-level reconnaissance, lateral movement, or authenticated Secret retrieval; this is the only path from application compromise to credential exfiltration via the Kubernetes API, since unauthenticated requests are rejected | remote_auth | openshift_api | openshift_secrets, openshift_route_config, inference_api_credentials | high | rare | partially_mitigated | OpenShift restricts default service account permissions; SecurityContextConstraints (SCCs) limit pod capabilities | Service account token auto-mounted at /var/run/secrets/kubernetes.io/serviceaccount unless `automountServiceAccountToken: false` is set; Kubernetes API is reachable from any pod at kubernetes.default.svc; if the service account has `get secrets` permission, this token provides authenticated access to all Secrets in the namespace |
| T12 | SQL injection via database interaction leads to data exfiltration or modification | remote_unauth | POST /analyze, all database-writing endpoints | database_records, postgresql_credentials | high | very_rare | mitigated | SQLAlchemy ORM with parameterized queries throughout; no raw SQL with user input | Only `text("SELECT 1")` used for health check |
| T9 | Attacker leverages long-running /analyze requests (up to 600s each) to tie up all Gunicorn workers, causing complete denial of service | remote_unauth | POST /analyze | service_availability | medium | possible | partially_mitigated | 16 workers; agent_timeout=600s; Gunicorn worker timeout=600s | No per-client connection limits; no request queue depth limits |
| T14 | Build log content (containing hostnames, internal paths, build environment details, and package metadata) is transmitted to an external cloud inference provider; compromise or misconfiguration of the provider, or overly broad data retention policies, exposes internal infrastructure details to a third party | supply_chain | external_inference_endpoint | build_log_content | medium | possible | partially_mitigated | PII sanitization removes emails, GPG fingerprints, RSA keys, and public key identifiers before sending; Drain3 extracts only representative snippets, reducing volume; TLS in transit | No controls on what the inference provider retains or how it processes the data beyond the provider's own terms of service; full snippet content is sent including hostnames and paths |
| T11 | Attacker sends oversized or specially crafted zip archives via GitLab artifacts, causing excessive memory consumption or zip bomb decompression | remote_unauth / remote_auth | POST /webhook/gitlab/job_events | service_availability | medium | rare | partially_mitigated | Content-Length check on artifacts zip; `zipfile.open()` per entry (no `extractall()`); max_artifact_size configurable | No zip compression ratio check; on-disk TemporaryFile for zip download, individual entries decompressed into memory via `zipfile.open().read()` |
| T13 | Attacker enumerates Koji task analysis results via sequential task ID guessing on the GET endpoint; when the Koji instance's token list is empty, the endpoint is unauthenticated (short-circuit `and` skips the check), reducing the barrier to enumeration | remote_unauth / remote_auth | GET /analyze/rpmbuild/koji/{instance}/{task_id} | llm_analysis_responses | low | possible | partially_mitigated | Koji token required only when `tokens` list is non-empty; task_id is integer path parameter | `if koji_instance_config.tokens and x_koji_token not in ...` — empty list bypasses auth (server.py:299,352); sequential integer IDs are predictable |

## 5. Deprioritized

| threat | reason |
|---|---|
| Physical access to server hardware | Out of scope; server is deployed on managed infrastructure (AWS or OpenShift cluster) with standard physical security controls |
| Direct attacks on local inference backend (vLLM/llama.cpp) | Applies only to podman-compose deployments with local inference; backends are on a private container network, not exposed to the internet. When using external cloud inference, the provider's security posture is covered by T14 |
| PostgreSQL network attacks | In podman-compose deployments, PostgreSQL port is published to the host via `ports:` mapping (all compose variants); access depends on host firewall and network configuration. In OpenShift, database is behind NetworkPolicies within the namespace. Risk is deployment-specific and mitigated by host-level network controls |
| DNS rebinding against SSRF protection | SSRFProtectedResolver checks resolved IPs at connect time, mitigating basic DNS rebinding; advanced TOCTOU attacks require precise timing and are impractical against this service |
| Malicious prompt templates on disk | Prompt templates are loaded from operator-managed files; an attacker with write access to these files already has sufficient access to compromise the system directly |
| CORS-based cross-origin attacks | No CORSMiddleware is configured; without it, FastAPI returns responses without `Access-Control-Allow-Origin` headers, so browsers block client-side JavaScript from reading responses (Same-Origin Policy). The server still processes and returns the request — the browser enforces the restriction, not the server. For this JSON API (`Content-Type: application/json`), browsers send a preflight OPTIONS request first, and without CORS headers in the preflight response, the actual request is not sent. This is the secure default for an API-only service |
| Supply chain compromise of generic Python dependencies | Generic risk applicable to any Python project. General controls: poetry.lock version pinning, hash-based dependency verification |

## 6. Open questions

- Should the Koji callback URL be restricted to an allowlist of known Koji instance domains rather than accepting arbitrary URLs?
- Are the GitLab API tokens scoped with minimum necessary permissions (e.g., read-only for project/job data, write-only for MR comments)?
- What is the intended behavior when `webhook_secrets` is empty for a GitLab instance in production? The current code treats all requests as authorized.
- Is there a process for verifying integrity of models pulled from HuggingFace before deployment?
- The `packages` code default is an empty list (deny all), but the shipped config file sets `packages: [".*"]` (all packages eligible for MR comments) — is this intentional for production, or should it be restricted to specific package namespaces?
- Is `automountServiceAccountToken: false` set on the application pod spec in OpenShift, or does the pod carry an unnecessary service account token?
- Are OpenShift NetworkPolicies configured to restrict pod egress to only the required destinations (inference provider, GitLab, Koji, PostgreSQL), or is egress unrestricted?
- What is the data retention and processing policy of the external cloud inference provider? Does it comply with the data handling requirements for the build log content being analyzed?

## 7. Provenance

- mode: bootstrap-then-interview
- date: 2026-07-30
- target: https://github.com/fedora-copr/logdetective @ df08181
- inputs: source code review, pyproject.toml dependency analysis, git commit history (security-related commits), docker-compose and deployment configuration, OpenShift deployment context (Routes, Secrets, NetworkPolicies, RBAC), inference service integration, THREAT_MODEL_README.md format specification
- owner: unset

## 8. Recommended mitigations

| mitigation | threat_ids | closes_class | effort |
|---|---|---|---|
| Add per-IP or per-token rate limiting middleware (e.g., slowapi) to `/analyze` and webhook endpoints | T5, T9 | partial | M |
| Implement model integrity verification by pinning expected SHA-256 hashes of GGUF model files and verifying after download | T2 | yes | M |
| Treat log text as untrusted data: strengthen delimitation of user-supplied content from system instructions beyond chat-API role separation (e.g., XML-tag wrapping or sentinel boundaries within user messages); extend `AgentResponse` schema validation to the CLI path (server path already validates via `expected_output=AgentResponse` and `model_validate()`); apply content-level filtering or a secondary classifier as defense in depth — note that tool use cannot be disabled on the server path since extraction tools are architecturally essential, but tools are sandboxed to extraction logic only | T6 | partial | M |
| Implement per-client concurrent request limits or a request queue with bounded depth to prevent worker exhaustion from slow requests | T5, T9 | partial | M |
| Set `automountServiceAccountToken: false` on the application pod spec to prevent unnecessary Kubernetes API access from a compromised pod | T16 | yes | S |
| Apply least-privilege RBAC: ensure the application's service account has no `get`/`list` permission on Secrets; restrict namespace access to operators only | T15 | yes | S |
| Establish a data processing agreement with the external inference provider covering retention limits, access controls, and breach notification for build log content | T14 | yes | M |
| Add zip decompression ratio check and per-entry size limit to prevent zip bomb attacks via GitLab artifacts; consider streaming decompression instead of reading entire entries into memory | T11 | partial | S |
| Require non-empty Koji token list for production deployments; add a startup warning or validation error when `tokens` is empty in non-development environments | T13 | yes | S |
| Configure egress NetworkPolicies in OpenShift to restrict outbound traffic from the application pod to only required destinations (inference provider, GitLab, Koji, PostgreSQL) | T16 | partial | M |
