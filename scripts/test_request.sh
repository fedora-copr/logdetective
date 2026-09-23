#!/bin/bash

# Smoke-test the asynchronous Koji API against configured Koji task URLs.
# This assumes a local server without API or Koji authentication enabled.

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8080}"
BASE_URL="${BASE_URL%/}"
KOJI_INSTANCE="${KOJI_INSTANCE:-fedora}"
MAX_POLLS="${MAX_POLLS:-120}"
RESULTS_DIR="${RESULTS_DIR:-./test_results/$(date +%Y%m%d_%H%M%S)}"
SERVER_CONTAINER="${SERVER_CONTAINER:-logdetective_server_1}"
WORKER_CONTAINER="${WORKER_CONTAINER:-logdetective_worker_1}"

# Provide list of URLS to Koji tasks
TASK_URLS=(
    # "https://kojipkgs.fedoraproject.org//work/tasks/9672/150349672/"
    # "https://kojipkgs.fedoraproject.org//work/tasks/1915/150341915/"
)
COMPLETION_REQUEST_ID="$(< /proc/sys/kernel/random/uuid)"
CANCELLATION_REQUEST_ID="$(< /proc/sys/kernel/random/uuid)"
TEST_START="$(date --iso-8601=seconds)"

mkdir -p "$RESULTS_DIR"

task_id_from_url() {
    local url="${1%/}"
    local task_id="${url##*/}"

    [[ "$task_id" =~ ^[1-9][0-9]*$ ]] || {
        echo "Unable to extract a positive task ID from $1" >&2
        return 1
    }
    echo "$task_id"
}

[ "${#TASK_URLS[@]}" -ge 2 ] || {
    echo "At least two Koji task URLs are required" >&2
    exit 1
}

COMPLETION_TASK_ID="$(task_id_from_url "${TASK_URLS[0]}")"
CANCELLATION_TASK_ID="$(task_id_from_url "${TASK_URLS[1]}")"

header_value() {
    awk -v name="$2" '
        BEGIN { IGNORECASE = 1 }
        $0 ~ "^" name ":" {
            sub("^[^:]+:[[:space:]]*", "")
            sub("\\r$", "")
            print
            exit
        }
    ' "$1"
}

submit_koji() {
    local request_id="$1"
    local task_id="$2"
    local name="$3"
    local status location

    status="$(curl --silent --show-error \
        --request POST \
        --header "Content-Type: application/json" \
        --dump-header "$RESULTS_DIR/${name}.headers" \
        --output "$RESULTS_DIR/${name}.json" \
        --write-out '%{http_code}' \
        --data "{\"id\":\"$request_id\",\"kojiInstance\":\"$KOJI_INSTANCE\",\"taskId\":$task_id}" \
        "$BASE_URL/analyze/rpmbuild/koji")"

    [ "$status" = 202 ] || {
        jq . "$RESULTS_DIR/${name}.json" 2>/dev/null || \
            sed -n '1,20p' "$RESULTS_DIR/${name}.json"
        echo "Expected HTTP 202, received $status" >&2
        return 1
    }

    location="$(header_value "$RESULTS_DIR/${name}.headers" Location)"
    [ -n "$location" ] || {
        echo "Submission did not return a Location header" >&2
        return 1
    }
    [ -n "$(header_value "$RESULTS_DIR/${name}.headers" Retry-After)" ] || {
        echo "Submission did not return a Retry-After header" >&2
        return 1
    }
    [ "$(jq --raw-output .id "$RESULTS_DIR/${name}.json")" = "$request_id" ]

    echo "$location"
}

poll_until() {
    local location="$1"
    local expected="$2"
    local name="$3"
    local attempt status retry_after http_status

    for ((attempt = 1; attempt <= MAX_POLLS; attempt++)); do
        http_status="$(curl --silent --show-error \
            --dump-header "$RESULTS_DIR/${name}.headers" \
            --output "$RESULTS_DIR/${name}.json" \
            --write-out '%{http_code}' \
            "$location")"
        [ "$http_status" = 200 ] || {
            echo "Polling returned HTTP $http_status" >&2
            return 1
        }

        status="$(jq --raw-output .status "$RESULTS_DIR/${name}.json")"
        echo "Poll $attempt/$MAX_POLLS: $status"
        case "$status" in
            "$expected") return ;;
            done|error|cancelled)
                jq . "$RESULTS_DIR/${name}.json"
                echo "Expected $expected, received $status" >&2
                return 1
                ;;
        esac

        retry_after="$(header_value "$RESULTS_DIR/${name}.headers" Retry-After)"
        [ -n "$retry_after" ] || retry_after=5
        sleep "$retry_after"
    done

    echo "Task did not reach $expected after $MAX_POLLS polls" >&2
    return 1
}

echo "=== Test 1: submit two tasks concurrently ==="
submit_koji \
    "$COMPLETION_REQUEST_ID" "$COMPLETION_TASK_ID" completion_submit \
    > "$RESULTS_DIR/completion.location" &
COMPLETION_SUBMIT_PID=$!
submit_koji \
    "$CANCELLATION_REQUEST_ID" "$CANCELLATION_TASK_ID" cancellation_submit \
    > "$RESULTS_DIR/cancellation.location" &
CANCELLATION_SUBMIT_PID=$!

wait "$COMPLETION_SUBMIT_PID"
wait "$CANCELLATION_SUBMIT_PID"
read -r COMPLETION_LOCATION < "$RESULTS_DIR/completion.location"
read -r CANCELLATION_LOCATION < "$RESULTS_DIR/cancellation.location"

echo "=== Test 2: idempotent task retry ==="
RETRY_LOCATION="$(submit_koji \
    "$COMPLETION_REQUEST_ID" "$COMPLETION_TASK_ID" completion_retry)"
[ "$RETRY_LOCATION" = "$COMPLETION_LOCATION" ] || {
    echo "Retry returned a different task location" >&2
    exit 1
}

echo "=== Test 3: conflicting request ID ==="
CONFLICT_STATUS="$(curl --silent --show-error \
    --request POST \
    --header "Content-Type: application/json" \
    --output "$RESULTS_DIR/conflict.json" \
    --write-out '%{http_code}' \
    --data "{\"id\":\"$COMPLETION_REQUEST_ID\",\"kojiInstance\":\"$KOJI_INSTANCE\",\"taskId\":$CANCELLATION_TASK_ID}" \
    "$BASE_URL/analyze/rpmbuild/koji")"
[ "$CONFLICT_STATUS" = 409 ] || {
    echo "Expected HTTP 409, received $CONFLICT_STATUS" >&2
    exit 1
}

echo "=== Test 4: cancel one concurrently submitted task ==="
CANCEL_STATUS="$(curl --silent --show-error \
    --request DELETE \
    --output "$RESULTS_DIR/cancellation_request.json" \
    --write-out '%{http_code}' \
    "$CANCELLATION_LOCATION")"
case "$CANCEL_STATUS" in
    200|202) ;;
    *)
        echo "Expected cancellation HTTP 200 or 202, received $CANCEL_STATUS" >&2
        exit 1
        ;;
esac
poll_until "$CANCELLATION_LOCATION" cancelled cancellation_result

echo "=== Test 5: wait for the other task result ==="
poll_until "$COMPLETION_LOCATION" done completion_result
jq --exit-status \
    ".result.task_id == $COMPLETION_TASK_ID and (.result.response.explanation | length > 0)" \
    "$RESULTS_DIR/completion_result.json" >/dev/null

podman logs --since "$TEST_START" "$SERVER_CONTAINER" \
    > "$RESULTS_DIR/server.log" 2>&1 || true
podman logs --since "$TEST_START" "$WORKER_CONTAINER" \
    > "$RESULTS_DIR/worker.log" 2>&1 || true

echo
echo "All asynchronous API tests passed. Results saved to $RESULTS_DIR"
echo "Completed task result:"
jq '.result' "$RESULTS_DIR/completion_result.json"
echo "Cancelled task result:"
jq . "$RESULTS_DIR/cancellation_result.json"
