from flytekit.sdk.tasks import dynamic_task, python_task, inputs
from flytekit.sdk.types import Types
from flytekit.sdk.workflow import Input, workflow_class


@python_task
def my_sub_task(wf_params):
    wf_params.logging.warn("hello world")


@inputs(array_size=Types.Integer)
@dynamic_task
def custom_array_size(wf_params, array_size):
    for i in range(0, array_size):
        yield my_sub_task()


@workflow_class
class CustomArraySizeWorkflow:
    array_size = Input(Types.Integer, default=3)
    arr = custom_array_size(array_size=array_size)
