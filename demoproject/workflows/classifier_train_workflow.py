import os
import ujson
import math
import random

from flytekit.sdk.workflow import workflow_class, Output, Input
from flytekit.sdk.types import Types
from flytekit.sdk.tasks import python_task, dynamic_task, inputs, outputs
from flytekit.common import utils as flytekit_utils

from models.classifier.resnet50.train_tasks import train_on_datasets
from utils.flyte_utils.fetch_executions import fetch_workflow_latest_execution

SERVICE_NAME = "flytekubecondemo2019"
DATAPREP_WORKFLOW_NAME = "workflows.data_preparation_workflow.DataPreparationWorkflow"
DEFAULT_SERVICE_INSTANCE = "development"

"""
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
"""
DEFAULT_VALIDATION_DATA_RATIO=0.2


def split_training_validation_streams(labeled_streams, validation_data_ratio):
    n_validation_streams = {
        c: int(math.ceil(len(labeled_streams[c]) * validation_data_ratio)) for c in labeled_streams.keys()
    }
    for _, s in labeled_streams.items():
        random.shuffle(s)

    validation_streams = {c: labeled_streams[c][:n_validation_streams[c]] for c in labeled_streams.keys()}
    training_streams = {c: labeled_streams[c][n_validation_streams[c]:] for c in labeled_streams.keys()}

    return training_streams, validation_streams


@inputs(
    training_validation_config_path=Types.String,  # The path to a json file listing the streams needed for training, and other parameters
    streams_metadata_path=Types.String,  # The path to a json file listing the metadata (e.g., class) of each stream
    validation_data_ratio=Types.Float,
)
@outputs(
    training_clean_mpblob=Types.MultiPartBlob,
    training_dirty_mpblob=Types.MultiPartBlob,
    validation_clean_mpblob=Types.MultiPartBlob,
    validation_dirty_mpblob=Types.MultiPartBlob,
)
@python_task(cache=True, cache_version="1", memory_request="500Mi")
def rearrange_data(
        wf_params,
        training_validation_config_path,
        streams_metadata_path,
        validation_data_ratio,
        training_clean_mpblob,
        training_dirty_mpblob,
        validation_clean_mpblob,
        validation_dirty_mpblob,
):
    # Get the latest execution of the data_prep_workflow
    latest_dataprep_wf_execution = fetch_workflow_latest_execution(
        service_name=SERVICE_NAME,
        workflow_name=DATAPREP_WORKFLOW_NAME,
        SERVICE_INSTANCE=DEFAULT_SERVICE_INSTANCE,
    )

    available_streams_mpblobs = latest_dataprep_wf_execution.outputs["selected_frames_mpblobs"]
    available_streams_names = latest_dataprep_wf_execution.outputs["selected_frames_stream_names"]

    # Download the config file and metadata
    training_validation_config_blob = Types.Blob.fetch(remote_path=training_validation_config_path)
    training_validation_config_blob.download()
    config = ujson.load(training_validation_config_blob.local_path)

    streams_metadata_blob = Types.Blob.fetch(remote_path=streams_metadata_path)
    streams_metadata_blob.download()
    streams_metadata = ujson.load(streams_metadata_blob.local_path)

    all_streams = streams_metadata.get("streams", {})
    selections = config.get("train_validation_datasets", {})
    training_validation_streams = [all_streams[s] for s in selections.keys()]

    # Splitting the set of streams into validation and training
    streams = {
        "clean": [s for s in training_validation_streams if s["class"] == "clean"],
        "dirty": [s for s in training_validation_streams if s["class"] == "dirty"],
    }
    training_streams, validation_streams = split_training_validation_streams(streams, validation_data_ratio)

    # Download multipartblobs to the target folders and then upload it
    with flytekit_utils.AutoDeletingTempDir("training") as training_dir:
        for label in streams.keys():
            output_dir = os.path.join(training_dir, label)

            for stream in training_streams[label]:
                idx = available_streams_names.index(stream)
                mpblob = available_streams_mpblobs[idx]
                mpblob.download(local_path=output_dir)

            if label == "clean":
                training_clean_mpblob.set(output_dir)
            elif label == "dirty":
                training_dirty_mpblob.set(output_dir)

    with flytekit_utils.AutoDeletingTempDir("validation") as validation_dir:
        for label in streams.keys():
            output_dir = os.path.join(validation_dir, label)

            for stream in validation_streams[label]:
                idx = available_streams_names.index(stream)
                mpblob = available_streams_mpblobs[idx]
                mpblob.download(local_path=output_dir)

            if label == "clean":
                validation_clean_mpblob.set(output_dir)
            elif label == "dirty":
                validation_dirty_mpblob.set(output_dir)


#@workflow_class
class ClassifierTrainWorkflow:

    training_validation_config_path = Input(Types.String, required=True)
    streams_metadata_path = Input(Types.String, required=True)
    validation_data_ratio = Input(Types.Float, default=DEFAULT_VALIDATION_DATA_RATIO)

    rearrange_data_task = rearrange_data(
        training_validation_config_path=training_validation_config_path,
        streams_metadata_path=streams_metadata_path,
        validation_data_ratio=validation_data_ratio,
    )

    train_on_datasets_task = train_on_datasets(
        training_clean_mpblob=rearrange_data_task.outputs.training_clean_mpblob,
        training_dirty_mpblob=rearrange_data_task.outputs.training_dirty_mpblob,
        validation_clean_mpblob=rearrange_data_task.outputs.validation_clean_mpblob,
        validation_dirty_mpblob=rearrange_data_task.outputs.validation_dirty_mpblob,
    )

    trained_models = Output(train_on_datasets_task.outputs.model_blobs, sdk_type=[Types.Blob])
    model_file_names = Output(train_on_datasets_task.outputs.model_files_names, sdk_type=[Types.String])

    # ------------------------------------------------------------
    """
    interactive_validate_model_config_task = interactive_validate_model_config(
        model_config_path=model_config_path
    )

    download_and_prepare_datasets_task = download_and_prepare_datasets(
        model_config_string=interactive_validate_model_config_task.outputs.model_config_string
    )

    train_on_datasets_task = train_on_datasets(
        train_data_mpblobs=download_and_prepare_datasets_task.outputs.train_data_mpblobs,
        validation_data_mpblobs=download_and_prepare_datasets_task.outputs.validate_zips_out,
        model_config_string=interactive_validate_model_config_task.outputs.model_config_string,
        model_output_path=model_output_path,
    )
    """