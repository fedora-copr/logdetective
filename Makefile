include env_file

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

server-down:
	$(COMPOSE_ENGINE) -f docker-compose-dev.yaml down

# WARNING: This target will start postgres and run migrations
# run alembic revision in another pod
alembic-generate-revision:
	@echo "Building server image..."
	$(COMPOSE_ENGINE) -f docker-compose-dev.yaml build server

	@echo "Starting postgres..."
	$(COMPOSE_ENGINE) -f docker-compose-dev.yaml up -d postgres

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
		--network logdetective_default \
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
		--network logdetective_default \
		localhost/logdetective/server:latest \
		bash -c "cd /src && python -m alembic revision -m \"$(CHANGE)\" --autogenerate"

	$(CONTAINER_ENGINE) rm logdetective_postgres_1 --force

# Download mermerd from:
# https://github.com/KarnerTh/mermerd/releases/download/v0.12.0/mermerd_0.12.0_linux_arm64.tar.gz
generate-db-diagram: server-up
	scripts/await_psql
	mermerd -c postgresql://$(POSTGRESQL_USER):$(POSTGRESQL_PASSWORD)@localhost:5432 -s public --useAllTables -o alembic/diagram.mmd
	echo "# ER diagram" > alembic/er_diagram.md
	echo -e '```mermaid' >> alembic/er_diagram.md
	cat alembic/diagram.mmd >> alembic/er_diagram.md
	echo '```' >> alembic/er_diagram.md
