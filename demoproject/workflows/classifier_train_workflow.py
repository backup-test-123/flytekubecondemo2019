from flytekit.sdk.workflow import workflow_class, Output, Input
from flytekit.sdk.tasks import python_task, inputs, outputs

DEFAULT_MODEL_CONFIG_PATH=
DEFAULT_MODEL_OUTPUT_PATH=


@inputs(
    train_zips=[Types.Blob],
    validation_zips=[Types.Blob],
    model_config_string=Types.String,
    model_output_path=Types.String,
)
@outputs(
    model_blobs=[Types.Blob],
    model_files_names=[Types.String],
)
@python_task(cache_version="1")
def train_on_datasets(
    wf_params,
    train_zips,
    model_config_string,
    model_output_path,
    model_blobs,
    model_files_names,
):

@workflow_class
class ClassifierTrainWorkflow:

    model_config_path = Input(Types.Blob, default=DEFAULT_MODEL_CONFIG_PATH)
    model_output_path = Input(Types.String, default=DEFAULT_MODEL_OUTPUT_PATH)

    validate_model_config_task = validate_model_config(
        model_config_path=model_config_path
    )

    download_and_prepare_task = download_and_prepare_datasets(
        model_config_string=validate_model_config_task.outputs.model_config_string
    )

    train_task = train_on_datasets(
        train_zips=download_and_prepare_task.outputs.train_zips_out,
        validation_zips=download_and_prepare_task.outputs.validate_zips_out,
        model_config_string=validate_model_config_task.outputs.model_config_string,
        model_config_path=model_output_path,
    )