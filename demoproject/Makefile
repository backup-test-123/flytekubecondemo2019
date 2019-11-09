export IMAGE_NAME=flytekubecondemo2019
PROJECT=flytekubecondemo2019
DOMAIN=development
VERSION=$(shell ./version.sh)

.PHONY: serialize
serialize:
	pyflyte -p ${PROJECT} -d ${DOMAIN} --config /app/sandbox.config serialize workflows

.PHONY: register_sandbox_in_container
register_sandbox_in_container:
	env
	pyflyte -p ${PROJECT} -d ${DOMAIN} --config /app/sandbox.config register workflows

.PHONY: register_sandbox
register_sandbox: docker_push
	kubectl delete job register || true
	kubectl create job register --image=docker.io/lyft/${IMAGE_NAME}:${VERSION} -- /opt/flytekit_venv make register_sandbox_in_container

.PHONY: execute_sandbox
execute_sandbox:
	flyte-cli -h localhost:30081 -p flytekubecondemo2019 -d development --insecure execute-launch-plan --urn lp:flytekubecondemo2019:development:workflows.data_preparation_workflow.DataPreparationWorkflow:${VERSION} -r demo -- video_external_paths='["s3://lyft-modelbuilder/metadata/_FlyteKubeconDemo2019Dataset/1537396662_cam-rgb-1.avi","s3://lyft-modelbuilder/metadata/_FlyteKubeconDemo2019Dataset/1538521877_cam-rgb-1.avi","s3://lyft-modelbuilder/metadata/_FlyteKubeconDemo2019Dataset/1538523916_cam-rgb-2.avi"]'

.PHONY: register_staging_in_container
register_staging_in_container:
	pyflyte -p ${PROJECT} -d ${DOMAIN} --config /app/staging.config register workflows

.PHONY: register_staging
register_staging: docker_push
	docker run docker.io/lyft/${IMAGE_NAME}:${VERSION} /opt/flytekit_venv make register_staging_in_container

.PHONY: execute_staging
execute_staging:
	# flyte-cli -h flyte-staging.lyft.net -p flytekubecondemo2019 -d development execute-launch-plan --urn lp:flytekubecondemo2019:development:workflows.data_preparation_workflow.DataPreparationWorkflow:${VERSION} -r demo -- video_external_paths='["s3://lyft-modelbuilder/metadata/_FlyteKubeconDemo2019Dataset/1537396662_cam-rgb-1.avi","s3://lyft-modelbuilder/metadata/_FlyteKubeconDemo2019Dataset/1538521877_cam-rgb-1.avi","s3://lyft-modelbuilder/metadata/_FlyteKubeconDemo2019Dataset/1538523916_cam-rgb-2.avi"]'
	flyte-cli -h flyte-staging.lyft.net -p flytekubecondemo2019 -d development execute-launch-plan --urn lp:flytekubecondemo2019:development:workflows.data_preparation_workflow.DataPreparationWorkflow:${VERSION} -r demo -- video_external_paths='["s3://lyft-modelbuilder/metadata/_FlyteKubeconDemo2019Dataset/1538521877_cam-rgb-1.avi","s3://lyft-modelbuilder/metadata/_FlyteKubeconDemo2019Dataset/1538523916_cam-rgb-2.avi"]'

.PHONY: docker_build
docker_build:
	scripts/docker_build.sh

.PHONY: docker_push
docker_push:
	REGISTRY=docker.io/lyft scripts/docker_build.sh
