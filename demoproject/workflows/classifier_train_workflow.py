from flytekit.sdk.workflow import workflow_class, Output, Input
from flytekit.sdk.types import Types
from flytekit.sdk.tasks import python_task, inputs, outputs
from flytekit.contrib.notebook import python_notebook

DEFAULT_MODEL_CONFIG_PATH=
DEFAULT_MODEL_OUTPUT_PATH=

interactive_validate_model_config = python_notebook(
    notebook_path="../models/classifier/resnet50/validate_model_config.ipynb",
    inputs={
        "model_config_path": Types.Blob,
    },
    outputs={
        "model_config_string": Types.String,
    },
    cache=True,
    cache_version="1",
)

interactive_download_and_prepare_datasets = python_notebook(
    notebook_path="../models/classifier/resnet50/download_and_prepare_datasets.ipynb",
    inputs={
        "data_session_id": Types.String,
        "sub_path": Types.String,
        "stream_name": Types.String,
        "stream_config_string": Types.String,
    },
    outputs={
        "output_zip": Types.Blob,
    },
    memory_request="16Gi",
    cpu_request="8",
    cache=True,
    cache_version="1",
)

interactive_train_on_datasets = python_notebook(
    notebook_path="../models/classifier/resnet50/train.ipynb",
    inputs={
        "train_mpblobs": [Types.MultiPartBlob],
        "validate_mpblobs": [Types.MultiPartBlob],
        "model_config_string": Types.String,
        "model_output_path": Types.String,
    },
    outputs={
        "model_blobs": [Types.Blob],
        "model_file_names": [Types.String],
    },
    memory_request="64Gi",
    gpu_request="1",
    cache=True,
    cache_version="1",
)

@workflow_class
class ClassifierTrainWorkflow:

    model_config_path = Input(Types.Blob, default=DEFAULT_MODEL_CONFIG_PATH)
    model_output_path = Input(Types.String, default=DEFAULT_MODEL_OUTPUT_PATH)

    validate_model_config_task = interactive_validate_model_config(
        model_config_path=model_config_path
    )

    download_and_prepare_task = interactive_download_and_prepare_datasets(
        model_config_string=validate_model_config_task.outputs.model_config_string
    )

    train_task = interactive_train_on_datasets(
        train_zips=download_and_prepare_task.outputs.train_zips_out,
        validation_zips=download_and_prepare_task.outputs.validate_zips_out,
        model_config_string=validate_model_config_task.outputs.model_config_string,
        model_config_path=model_output_path,
    )