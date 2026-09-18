#!/usr/bin/bash
set -eux

if [ "$ENV" == "production" ]; then
    conf="/src/server/gunicorn-prod.config.py"
elif [ "$ENV" == "http_production" ]; then
    conf="/src/server/gunicorn-prod-http.config.py"
else
    conf="/src/server/gunicorn-dev.config.py"
fi

echo "Starting application server..."
exec python -m gunicorn -c "$conf" logdetective.server:app
