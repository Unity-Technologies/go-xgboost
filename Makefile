build-docker:
	./scripts/build_docker.sh
.PHONY: build-docker

test:
	./scripts/docker_run.sh go test ./...
.PHONY: test