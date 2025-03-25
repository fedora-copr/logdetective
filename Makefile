COMPOSE_ENGINE ?= $(shell command -v podman-compose 2> /dev/null || echo docker-compose)

.PHONY: server

# rebuild server image (for compose) with new and updated dependencies
rebuild-server:
	$(COMPOSE_ENGINE) build --no-cache server
