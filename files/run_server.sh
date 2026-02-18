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

if [ "$ENV" == "production" ]; then
    conf="/src/server/gunicorn-prod.config.py"
else
    conf="/src/server/gunicorn-dev.config.py"
fi

echo "Starting application server..."
exec python -m gunicorn -c "$conf" logdetective.server.server:app
