#!/bin/bash
# Enable vector extension

set -eux

psql -v ON_ERROR_STOP=1 -d "${POSTGRESQL_DATABASE}" -c "CREATE EXTENSION IF NOT EXISTS vector;"
