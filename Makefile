export IMAGE_NAME=flytekubecondemo2019

docker_build:
	scripts/docker_build.sh

docker_push:
	REGISTRY=docker.io/lyft scripts/docker_build.sh
