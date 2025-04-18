include .env

CONTAINER_ENGINE ?= $(shell command -v podman 2> /dev/null || echo docker)
COMPOSE_ENGINE ?= $(shell command -v podman-compose 2> /dev/null || echo docker-compose)
MY_ID ?= $(shell id -u)
CHANGE ?= New migration

.PHONY: server-up rebuild-server alembic-generate-revision

# rebuild server image (for compose) with new and updated dependencies
rebuild-server:
	$(COMPOSE_ENGINE) -f docker-compose-dev.yaml build --no-cache server

server-up:
	$(COMPOSE_ENGINE) -f docker-compose-dev.yaml up --build --force-recreate -d server

# WARNING: This target will start up a new server
# and shut it down when the operation completes
# run alembic revision in another pod
alembic-generate-revision: server-up
	@echo "Waiting for server to start and invoking alembic upgrade head..."
	sleep 3

	@echo "Checking if database is ready..."
	$(CONTAINER_ENGINE) run --rm --network logdetective_default \
		quay.io/sclorg/postgresql-15-c9s \
		pg_isready -h postgres -U $(POSTGRESQL_USER) -d $(POSTGRESQL_DATABASE) \
		|| (echo "Database not ready -h postgres -U $(POSTGRESQL_USER) -d $(POSTGRESQL_DATABASE)"; exit 1)

	$(CONTAINER_ENGINE) run --rm -ti --user $(MY_ID) --uidmap=$(MY_ID):0:1 --uidmap=0:1:999 \
		-e POSTGRESQL_USER=$(POSTGRESQL_USER) \
		-e POSTGRESQL_PASSWORD=$(POSTGRESQL_PASSWORD) \
		-e POSTGRESQL_HOST=postgres \
		-e POSTGRESQL_DATABASE=$(POSTGRESQL_DATABASE) \
		-v $(PWD)/alembic:/src/alembic:rw,z \
		-v $(PWD)/alembic.ini:/src/alembic.ini:ro,z \
		--network logdetective_default \
		localhost/logdetective/server:latest \
		bash -c "cd /src && alembic revision -m \"$(CHANGE)\" --autogenerate"

	@echo "WARNING: Shutting down server..."
	$(COMPOSE_ENGINE) down server

# Download mermerd from:
# https://github.com/KarnerTh/mermerd/releases/download/v0.12.0/mermerd_0.12.0_linux_arm64.tar.gz
generate-db-diagram: server-up
	sleep 3
	mermerd -c postgresql://$(POSTGRESQL_USER):$(POSTGRESQL_PASSWORD)@localhost:5432 -s public --useAllTables -o alembic/diagram.mmd
	echo "# ER diagram" > alembic/er_diagram.md
	echo -e '```mermaid' >> alembic/er_diagram.md
	cat alembic/diagram.mmd >> alembic/er_diagram.md
	echo '```' >> alembic/er_diagram.md
