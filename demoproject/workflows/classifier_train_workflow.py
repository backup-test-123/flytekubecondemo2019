from flytekit.sdk.workflow import workflow_class, Output, Input
from flytekit.sdk.types import Types
from flytekit.sdk.tasks import python_task, inputs, outputs
from flytekit.contrib.notebook import python_notebook

DEFAULT_MODEL_CONFIG_PATH=
DEFAULT_MODEL_OUTPUT_PATH=

interactive_validate_model_config = python_notebook(notebook_path="../models/classifier/resnet50/train.ipynb")
interactive_download_and_prepare_datasets = python_notebook(notebook_path="")
interactive_train_on_dataset = python_notebook(notebook_path="../models/classifier/resnet50/train.ipynb")

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