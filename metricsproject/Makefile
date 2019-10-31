export IMAGE_NAME=flytekubecondemo2019_metrics
PROJECT=flytekubecondemo2019_metrics
DOMAIN=development
VERSION=$(shell ./version.sh)

serialize:
	pyflyte -p ${PROJECT} -d ${DOMAIN} --config /app/sandbox.config serialize tasks

register:
	pyflyte -p ${PROJECT} -d ${DOMAIN} --config /app/sandbox.config register tasks

.PHONY: docker_push
register_sandbox: docker_push
	kubectl delete job register || true
	kubectl create job register --image=docker.io/lyft/${IMAGE_NAME}:${VERSION} -- /opt/flytekit_venv make register

docker_build:
	scripts/docker_build.sh

docker_push:
	REGISTRY=docker.io/lyft scripts/docker_build.sh
