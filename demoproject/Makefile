export IMAGE_NAME=flytekubecondemo2019
PROJECT=flyteexamples
DOMAIN=development
VERSION=$(shell ./version.sh)

serialize:
	pyflyte -p ${PROJECT} -d ${DOMAIN} --config /app/sandbox.config serialize workflows

register:
	pyflyte -p ${PROJECT} -d ${DOMAIN} --config /app/sandbox.config register workflows

execute:
	pyflyte -p ${PROJECT} -d ${DOMAIN} --config /app/sandbox.config lp execute app.workflows.simple_workflow.SimpleWorkflow

docker_build:
	scripts/docker_build.sh

docker_push:
	REGISTRY=docker.io/lyft scripts/docker_build.sh
