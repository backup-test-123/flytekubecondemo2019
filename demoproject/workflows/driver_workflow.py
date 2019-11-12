
import ujson
from flytekit.sdk.workflow import workflow_class, Output, Input
from flytekit.sdk.types import Types
from flytekit.sdk.tasks import python_task, inputs, outputs
from flytekit.common import utils as flytekit_utils

from workflows.classifier_train_workflow import train_lp, DEFAULT_VALIDATION_DATA_RATIO, DEFAULT_TRAINING_VALIDATION_CONFIG_FILE
from workflows.classifier_evaluate_workflow import evaluate_lp
from workflows.data_preparation_workflow import data_prep

@workflow_class
class DriverWorkflow:
    streams_external_storage_prefix = Input(Types.String, required=True)
    streams_names = Input([Types.String], required=True)
    stream_extension = Input(Types.String, default="avi")

    streams_metadata_path = Input(Types.String, required=True)
    training_validation_config_json = Input(Types.Generic, default=ujson.loads(open(DEFAULT_TRAINING_VALIDATION_CONFIG_FILE).read()))
    validation_data_ratio = Input(Types.Float, default=DEFAULT_VALIDATION_DATA_RATIO)

    prepare = data_prep(
    	streams_external_storage_prefix=streams_external_storage_prefix,
        streams_names=streams_names,
        stream_extension=stream_extension)

    train = train_lp(
        streams_metadata_path=streams_metadata_path,
        training_validation_config_json=training_validation_config_json,
        validation_data_ratio=validation_data_ratio
    )

    evaluate = evaluate_lp(
        streams_metadata_path=streams_metadata_path,
        evaluation_config_json=training_validation_config_json,
        model=train.outputs.trained_models[1]
	)

    ground_truths = Output(evaluate.outputs.ground_truths, sdk_type=[Types.Integer])
    predictions = Output(evaluate.outputs.predictions, sdk_type=[Types.Integer])
