import os

from flytekit.common.workflow_execution import SdkWorkflowExecution
from flytekit.common.core import identifier as Flyte2Identifier
from flytekit.models.common import NamedEntityIdentifier as Flyte2NamedEntityIdentifier
from flytekit.models import filters as Flyte2Filters
from flytekit.models.admin import common as FlyteAdminCommon
from flytekit.clients import friendly as Flyte2FriendlyClient
from flytekit.configuration import platform as platform_config


def _get_flyte2_client() -> Flyte2FriendlyClient.SynchronousFlyteClient:
    try:
        return _get_flyte2_client.flyte2_client  # type: ignore
    except AttributeError:
        _get_flyte2_client.flyte2_client = Flyte2FriendlyClient.SynchronousFlyteClient(platform_config.URL)  # type: ignore
        return _get_flyte2_client.flyte2_client  # type: ignore


def _get_workflow_latest_sha(service_name, workflow_name, service_instance):
    filters = [Flyte2Filters.Filter.from_python_std(f'eq(workflow.name,{workflow_name})')]
    sort_by = 'desc(created_at)'
    wf_list, _ = _get_flyte2_client().list_workflows_paginated(
        identifier=Flyte2NamedEntityIdentifier(
            project=service_name,
            domain=service_instance,
            name=workflow_name
        ),
        limit=1,
        filters=filters,
        sort_by=FlyteAdminCommon.Sort.from_python_std(sort_by)
    )
    return wf_list[0].id.version


def fetch_workflow_latest_execution(service_name, workflow_name, service_instance):
    latest_execution_sha = _get_workflow_latest_sha(service_name, workflow_name, service_instance)

    return fetch_workflow_execution(
        service_name=service_name, service_instance=service_instance, exec_id=latest_execution_sha)

    """
    eid = Flyte2Identifier.WorkflowExecutionIdentifier(
        project=service_name, domain=service_instance, name=latest_execution_sha)

    # eid = Flyte2Identifier.WorkflowExecutionIdentifier.from_python_std(latest_execution_sha)

    wf_exec = SdkWorkflowExecution.fetch(project=eid.project, domain=eid.domain, name=eid.name)
    wf_exec.sync()

    return wf_exec
    """

def fetch_workflow_execution(service_name, service_instance, exec_id):
    eid = Flyte2Identifier.WorkflowExecutionIdentifier(
        project=service_name, domain=service_instance, name=exec_id)
    wf_exec = SdkWorkflowExecution.fetch(project=eid.project, domain=eid.domain, name=eid.name)
    wf_exec.sync()

    return wf_exec
