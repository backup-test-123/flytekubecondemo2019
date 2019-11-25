import time

from flytekit.sdk.tasks import dynamic_task, python_task, inputs, hive_task
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


@hive_task
def failing_hive_task(wf_params):
    return 'select reflect("java.lang.Threassd", "sleep", bigint(10000));'


@python_task
def my_other_sub_task(wf_params):
    wf_params.logging.warn("hello world 2")


@dynamic_task()
def failing_task(wf_params):
    yield my_sub_task()
    yield my_sub_task()
    yield my_sub_task()
    yield my_other_sub_task()
    yield failing_hive_task()
    yield failing_hive_task()
    yield failing_hive_task()


@python_task
def my_succeeding_sub_task(wf_params):
    time.sleep(1000)


@python_task
def my_failing_sub_task(wf_params):
    raise Exception("Failure expected")


@python_task
def my_failing_sub_task_2(wf_params):
    raise Exception("Failure expected 2")


@dynamic_task
def failing_dynamic_task2(wf_params):
    yield my_failing_sub_task_2()


@dynamic_task
def failing_dynamic_task(wf_params):
    yield my_failing_sub_task()
    yield my_failing_sub_task_2()
    yield my_succeeding_sub_task()
    yield failing_dynamic_task2()


@workflow_class
class FailingWorkflow:
    # failing_dynamic_task = failing_task()
    task = failing_dynamic_task()
