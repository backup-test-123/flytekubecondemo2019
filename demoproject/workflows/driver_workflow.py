
from flytekit.sdk.workflow import workflow_class, Output, Input
from flytekit.sdk.types import Types
from flytekit.sdk.tasks import python_task, inputs, outputs
from flytekit.common import utils as flytekit_utils

from classifier_train_workflow import DEFAULT_VALIDATION_DATA_RATIO

@workflow_class
class DriverWorkflow:
    streams_external_storage_prefix = Input(Types.String, required=True)
    streams_names = Input([Types.String], required=True)
    stream_extension = Input(Types.String, default="avi")

    streams_metadata_path = Input(Types.String, required=True)
    training_validation_config_json = Input(Types.Generic, default=ujson.loads(open(DEFAULT_TRAINING_VALIDATION_CONFIG_FILE).read()))
    validation_data_ratio = Input(Types.Float, default=DEFAULT_VALIDATION_DATA_RATIO)

    prepare = DataPreparationWorkflow(streams_external_storage_prefix=streams_external_storage_prefix,
        streams_names=streams_names,
        stream_extension=stream_extension)

    train = ClassifierTrainWorkflow.execute(
        streams_metadata_path=streams_metadata_path,
        training_validation_config_json=training_validation_config_json,
        validation_data_ratio=validation_data_ratio
    )

    evaluate = ClassifierEvaluateWorkflow.execute(
        streams_metadata_path=streams_metadata_path,
        evaluation_config_json=evaluation_config_json,
        model=train.outputs.trained_models[1]
	)

    ground_truths = Output(evaluate.outputs.ground_truths, sdk_type=[Types.Integer])
    predictions = Output(evaluate.outputs.predictions, sdk_type=[Types.Integer])
