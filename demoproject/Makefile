export IMAGE_NAME=flytekubecondemo2019
PROJECT=flytesnacks
DOMAIN=development
VERSION=$(shell ./version.sh)

serialize:
	pyflyte -p ${PROJECT} -d ${DOMAIN} --config /app/sandbox.config serialize workflows

register:
	pyflyte -p ${PROJECT} -d ${DOMAIN} --config /app/sandbox.config register workflows

.PHONY: docker_push
register_sandbox: docker_push
	kubectl create job register --image=docker.io/lyft/${IMAGE_NAME}:${VERSION} -- /opt/flytekit_venv make register

execute:
	pyflyte -p ${PROJECT} -d ${DOMAIN} --config /app/sandbox.config lp execute app.workflows.simple_workflow.SimpleWorkflow

docker_build:
	scripts/docker_build.sh

docker_push:
	REGISTRY=docker.io/lyft scripts/docker_build.sh
