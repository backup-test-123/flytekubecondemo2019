from __future__ import absolute_import

from flytekit.sdk.tasks import (
    python_task,
    inputs,
    outputs,
)
from flytekit.sdk.types import Types
from flytekit.sdk.workflow import workflow_class, Input, Output


@inputs(value_to_print=Types.Integer)
@outputs(out=Types.Integer)
@python_task
def add_one_and_print(workflow_parameters, value_to_print, out):
    added = value_to_print + 1
    workflow_parameters.logging.info("My printed value: {}".format(added))
    out.set(added)


@inputs(values_to_print=[Types.Integer])
@outputs(out=Types.Integer)
@python_task
def sum_non_none(workflow_parameters, values_to_print, out):
    added = 0
    for value in values_to_print:
        workflow_parameters.logging.info("Adding values: {}".format(value))
        if value is not None:
            added += value
    added += 1
    workflow_parameters.logging.info("My printed value: {}".format(added))
    out.set(added)


@workflow_class
class SimpleWorkflow(object):
    triggered_date = Input(Types.Datetime)
    print1a = add_one_and_print(value_to_print=3)
    print1b = add_one_and_print(value_to_print=101)
    print2 = sum_non_none(values_to_print=[print1a.outputs.out, print1b.outputs.out])
    print3 = add_one_and_print(value_to_print=print2.outputs.out)
    print4 = add_one_and_print(value_to_print=print3.outputs.out)
    final_value = Output(print4.outputs.out, sdk_type=Types.Integer)
