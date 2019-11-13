import ujson
from flytekit.common.tasks.task import SdkTask
from flytekit.sdk.tasks import python_task, inputs, outputs
from flytekit.sdk.types import Types
from flytekit.sdk.workflow import workflow_class, Output, Input
from workflows.classifier_evaluate_workflow import evaluate_lp
from workflows.classifier_train_workflow import train_lp, DEFAULT_VALIDATION_DATA_RATIO, \
    DEFAULT_TRAINING_VALIDATION_CONFIG_FILE
from workflows.data_preparation_workflow import data_prep

compute_confusion_matrix = SdkTask.fetch(
        project="kubecondemo2019-metrics",
        domain="development",
        name="demo_metrics.tasks.confusion_matrix.confusion_matrix",
        version="a2f96929b3e5e354b5848b7b4d7025547e8875e8",
    )


@inputs(models=[Types.Blob])
@outputs(second=Types.Blob)
@python_task(cache=True, cache_version="1")
def pick_second(wf_params, models, second):
    second.set(models[1])


@workflow_class
class DriverWorkflow:
    streams_external_storage_prefix = Input(Types.String, required=True)
    streams_names = Input([Types.String], required=True)
    stream_extension = Input(Types.String, default="avi")

    streams_metadata_path = Input(Types.String, required=True)
    training_validation_config_json = Input(Types.Generic,
                                            default=ujson.loads(open(DEFAULT_TRAINING_VALIDATION_CONFIG_FILE).read()))
    validation_data_ratio = Input(Types.Float, default=DEFAULT_VALIDATION_DATA_RATIO)

    prepare = data_prep(
        streams_external_storage_prefix=streams_external_storage_prefix,
        streams_names=streams_names,
        stream_extension=stream_extension)

    train = train_lp(
        available_streams_names=prepare.outputs.streams_names_out,
        available_streams_mpblobs=prepare.outputs.selected_frames_mpblobs,
        streams_metadata_path=streams_metadata_path,
        training_validation_config_json=training_validation_config_json,
        validation_data_ratio=validation_data_ratio
    )

    pick_second = pick_second(models=train.outputs.trained_models)

    evaluate = evaluate_lp(
        available_streams_names=prepare.outputs.streams_names_out,
        available_streams_mpblobs=prepare.outputs.selected_frames_mpblobs,
        streams_metadata_path=streams_metadata_path,
        evaluation_config_json=training_validation_config_json,
        model=pick_second.outputs.second,
        validation_data_ratio=validation_data_ratio
    )

    confusion_matrix_task = compute_confusion_matrix(
        y_true=evaluate.outputs.ground_truths,
        y_pred=evaluate.outputs.predictions,
        title="Confusion Matrix",
        normalize=True,
        classes=["dirty", "clean"],
    )

    ground_truths = Output(evaluate.outputs.ground_truths, sdk_type=[Types.Integer])
    predictions = Output(evaluate.outputs.predictions, sdk_type=[Types.Integer])
    confusion_matrix = Output(confusion_matrix_task.outputs.matrix, sdk_type=[[Types.Integer]])
    confusion_matrix_image = Output(confusion_matrix_task.outputs.visual, sdk_type=Types.Blob)
