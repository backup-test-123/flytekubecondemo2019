export IMAGE_NAME=flytekubecondemo2019
PROJECT=flytekubecondemo2019
DOMAIN=development
VERSION=$(shell ./version.sh)

.PHONY: serialize
serialize:
	pyflyte -p ${PROJECT} -d ${DOMAIN} --config /app/sandbox.config serialize workflows

.PHONY: register_sandbox_in_container
register_sandbox_in_container:
	pyflyte -p ${PROJECT} -d ${DOMAIN} --config /app/sandbox.config register workflows

.PHONY: docker_push
register_sandbox: docker_push
	kubectl delete job register || true
	kubectl create job register --image=docker.io/lyft/${IMAGE_NAME}:${VERSION} -- /opt/flytekit_venv make register_sandbox_in_container

.PHONY: register_staging_in_container
register_staging_in_container:
	pyflyte -p ${PROJECT} -d ${DOMAIN} --config /app/staging.config register workflows

.PHONY: docker_push
register_staging: docker_push
	docker run docker.io/lyft/${IMAGE_NAME}:${VERSION} /opt/flytekit_venv make register_staging_in_container

.PHONY: docker_build
docker_build:
	scripts/docker_build.sh

.PHONY: docker_push
docker_push:
	REGISTRY=docker.io/lyft scripts/docker_build.sh
