include env_file

CONTAINER_ENGINE ?= $(shell command -v podman 2> /dev/null || echo docker)
COMPOSE_ENGINE ?= $(shell command -v podman-compose 2> /dev/null || echo docker-compose)

COMPOSE_DEV = docker-compose-dev.yaml
COMPOSE_PROD = docker-compose-prod.yaml
COMPOSE_PROJECT = logdetective
COMPOSE_FOLDER = containers

CHANGE ?= New migration
COMPOSE_CMD = cd $(COMPOSE_FOLDER) && $(COMPOSE_ENGINE) --project-name $(COMPOSE_PROJECT)

MY_ID ?= $(shell id -u)

.PHONY: rebuild-server rebuild-postgres dev-up dev-down prod-up prod-down alembic-generate-revision run-mermerd generate-db-diagram

# rebuild server image (for compose) with new and updated dependencies
rebuild-server:
	$(COMPOSE_CMD) --file $(COMPOSE_DEV) build --no-cache server

rebuild-postgres:
	$(COMPOSE_CMD) --file $(COMPOSE_DEV) build --no-cache postgres

dev-up:
	$(COMPOSE_CMD) --file $(COMPOSE_DEV) up -d server

dev-down:
	$(COMPOSE_CMD) --file $(COMPOSE_DEV) down

prod-up:
	$(COMPOSE_CMD) --file $(COMPOSE_PROD) up -d

prod-down:
	$(COMPOSE_CMD) --file $(COMPOSE_PROD) down

# WARNING: This target will start postgres and run migrations
# run alembic revision in another pod
alembic-generate-revision:
	@echo "Building server image..."
	$(COMPOSE_CMD) --file $(COMPOSE_DEV) build server

	@echo "Starting postgres..."
	$(COMPOSE_CMD) --file $(COMPOSE_DEV) up -d postgres

	@echo "Waiting for postgres to be ready..."
	scripts/await_psql --skip-alembic

	@echo "Running existing migrations to head..."
	$(CONTAINER_ENGINE) run --rm --user $(MY_ID) --uidmap=$(MY_ID):0:1 --uidmap=0:1:999 \
		-e POSTGRESQL_USER=$(POSTGRESQL_USER) \
		-e POSTGRESQL_PASSWORD=$(POSTGRESQL_PASSWORD) \
		-e POSTGRESQL_HOST=postgres \
		-e POSTGRESQL_DATABASE=$(POSTGRESQL_DATABASE) \
		-v $(PWD)/logdetective:/src/logdetective:ro,z \
		-v $(PWD)/alembic:/src/alembic:ro,z \
		-v $(PWD)/alembic.ini:/src/alembic.ini:ro,z \
		--network $(COMPOSE_PROJECT)_default \
		localhost/logdetective/server:latest \
		bash -c "cd /src && python -m alembic upgrade head"

	@echo "Generating new revision..."
	$(CONTAINER_ENGINE) run --rm -ti --user $(MY_ID) --uidmap=$(MY_ID):0:1 --uidmap=0:1:999 \
		-e POSTGRESQL_USER=$(POSTGRESQL_USER) \
		-e POSTGRESQL_PASSWORD=$(POSTGRESQL_PASSWORD) \
		-e POSTGRESQL_HOST=postgres \
		-e POSTGRESQL_DATABASE=$(POSTGRESQL_DATABASE) \
		-v $(PWD)/logdetective:/src/logdetective:ro,z \
		-v $(PWD)/alembic:/src/alembic:rw,z \
		-v $(PWD)/alembic.ini:/src/alembic.ini:ro,z \
		--network $(COMPOSE_PROJECT)_default \
		localhost/logdetective/server:latest \
		bash -c "cd /src && python -m alembic revision -m \"$(CHANGE)\" --autogenerate"

	@echo "Cleaning up..."
	$(COMPOSE_CMD) --file $(COMPOSE_DEV) down

# Download mermerd from:
# https://github.com/KarnerTh/mermerd/releases/download/v0.12.0/mermerd_0.12.0_linux_arm64.tar.gz
generate-db-diagram: dev-up run-mermerd dev-down

run-mermerd:
	scripts/await_psql
	mermerd -c postgresql://$(POSTGRESQL_USER):$(POSTGRESQL_PASSWORD)@localhost:5432 -s public --useAllTables -o alembic/diagram.mmd
	echo "# ER diagram" > alembic/er_diagram.md
	echo -e '```mermaid' >> alembic/er_diagram.md
	cat alembic/diagram.mmd >> alembic/er_diagram.md
	echo '```' >> alembic/er_diagram.md
