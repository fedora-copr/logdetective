#!/usr/bin/bash
set -eux
echo "Running database migrations..."

# if all containers started at the same time, pg is definitely not ready to serve
# so let's try this for a few times
ATTEMPTS=7
n=0
while [[ $n -lt $ATTEMPTS ]]; do
  alembic-3 upgrade head && break
  n=$((n+1))
  sleep 2
done

# If the number of attempts was exhausted: the migration failed.
# Exit with an error.
if [[ $n -eq $ATTEMPTS ]]; then
    echo "Migration failed after $ATTEMPTS attempts. Exiting."
    exit 1
fi

echo "Starting application server..."
# --no-reload: doesn't work in a container - `PermissionError: Permission denied (os error 13) about ["/proc"]`
# command: fastapi dev /src/logdetective/server.py --host 0.0.0.0 --port $LOGDETECTIVE_SERVER_PORT --no-reload
# timeout set to 240 - 4 minutes should be enough for one LLM execution locally on a CPU
gunicorn -k uvicorn.workers.UvicornWorker --timeout 240 logdetective.server.server:app -b 0.0.0.0:${LOGDETECTIVE_SERVER_PORT}
