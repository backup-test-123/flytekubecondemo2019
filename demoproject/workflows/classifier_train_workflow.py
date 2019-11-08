import ujson
from flytekit.sdk.workflow import workflow_class, Output, Input
from flytekit.sdk.types import Types
from flytekit.sdk.tasks import python_task, dynamic_task, inputs, outputs
from flytekit.contrib.notebook import python_notebook

from models.classifier.resnet50.train_tasks import train_on_datasets
from utils.flyte_utils.fetch_executions import fetch_workflow_latest_execution

DEFAULT_MODEL_CONFIG_PATH=
DEFAULT_MODEL_OUTPUT_PATH=

SERVICE_NAME = "flytekubecondemo2019"
DATAPREP_WORKFLOW_NAME = "workflows.data_preparation_workflow.DataPreparationWorkflow"
DEFAULT_SERVICE_INSTANCE = "development"

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
"""


@inputs(
    target_streams=[Types.String],
    stream_config_string=Types.String,
)
@outputs(
    output_zip=Types.Blob,
)
@python_task(cache=True, cache_version="1", memory_request="4Gi")
def download_and_prepare_dataset_worker(
        target_streams,
        stream_config_string,
        output_zip
):
    """
    Take a dataset pair (collection_id, sub_path, and stream_name) and archive it all into 1 zip file
    so the training workflow has a quicker time downloading it all
    This also supports the sub-selection elements that exist in the config where we can sub-select a subset of
    the content in that dataset
    """

    dataset_config_json = ujson.loads(stream_config_string)
    tmp_folder = wf_params.working_directory.get_named_tempfile("input")
    output_zip_file_name = wf_params.working_directory.get_named_tempfile("output.zip")

    output_dir_path = Path(tmp_folder)
    if output_dir_path.exists():
        rmtree(tmp_folder)
    output_dir_path.mkdir(0o777, parents=True, exist_ok=False)

    s3_client = boto3.client("s3")
    bucket = BUCKET_NAME
    prefix = DATA_PATH_FORMAT.format(
        collection_id=data_session_id, sub_path=sub_path, stream_name=stream_name
    )

    # List all objects within a S3 bucket path
    start = dataset_config_json.get("start", 0)
    end = dataset_config_json.get("end", 100)
    every_n = dataset_config_json.get("every_n", 1)
    every_n_offset = dataset_config_json.get("every_n_offset", 0)

    dataset_keys = s3_list_contents_paginated(
        s3_client=s3_client, bucket=bucket, prefix=prefix
    )
    dataset_size = len(dataset_keys)
    # Loop through each file
    for i in range(0, dataset_size):
        # Get the file name
        file = dataset_keys[i]
        frame_id = should_include_frame_in_subset(
            frame_key_name=file["Key"],
            dataset_size=dataset_size,
            subset_start=start,
            subset_end=end,
            subset_every_n=every_n,
            subset_every_n_offset=every_n_offset,
        )
        if not frame_id:
            continue

        # download, with a more specific name to tmp
        new_file_name = (
            f"{tmp_folder}/Frame_{data_session_id}_{stream_name}_{frame_id:05d}.png"
        )
        print("downloading {} to {}".format(file["Key"], new_file_name))
        s3_client.download_file(bucket, file["Key"], new_file_name)

    zipdir(tmp_folder, output_zip_file_name)
    zip_blob = Types.Blob()
    with zip_blob as fileobj:
        with open(output_zip_file_name, mode="rb") as file:  # b is important -> binary
            fileobj.write(file.read())


@inputs(
    model_config_string=Types.String,
    validation_streams_ratio=Types.Float,
    metadata_path=Types.String,
)
@outputs(
    train_data_mpblobs=[Types.MultiPartBlob],
    validate_data_mpblobs=[Types.MultiPartBlob],
)
@dynamic_task(cache=True, cache_version="1")
def download_and_prepare_datasets(
    wf_params, model_config_string, train_zips_out, validate_zips_out
):
    model_config = ujson.loads(model_config_string)

    dataset model_config.get("training_validation_datasets", {})


    train_streams = flatten_session_sub_path_stream_tuple(
        model_config.get("train_datasets", {})
    )

    validation_streams = flatten_session_sub_path_stream_tuple(
        model_config.get("validation_datasets", {})
    )

    train_zips_out.set(
        [
            download_and_prepare_dataset_worker(
                data_session_id=data_stream_config.collection_id,
                sub_path=data_stream_config.sub_path,
                stream_name=data_stream_config.stream_name,
                stream_config_string=ujson.dumps(data_stream_config.stream_config),
            ).outputs.output_zip
            for data_stream_config in train_streams
        ]
    )
    validate_zips_out.set(
        [
            download_and_prepare_dataset_worker(
                data_session_id=data_stream_config.collection_id,
                sub_path=data_stream_config.sub_path,
                stream_name=data_stream_config.stream_name,
                stream_config_string=ujson.dumps(data_stream_config.stream_config),
            ).outputs.output_zip
            for data_stream_config in validation_streams
        ]
    )

DEFAULT_VALIDATION_DATA_RATIO=0.2





@inputs(
    training_validation_config_path=Types.String,  # The path to a json file listing the streams needed for training, and other parameters
    streams_metadata_path=Types.String,  # The path to a json file listing the metadata (e.g., class) of each stream
    validation_data_ratio=Types.Float,
)
@outputs(
    training_clean_streams_mpblob=Types.MultiPartBlob,
    training_dirty_streams_mpblob=Types.MultiPartBlob,
    validation_dirty_mpblob=Types.MultiPartBlob,
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
    # Create a multipartblob for training_clean



@workflow_class
class ClassifierTrainWorkflow:

    training_validation_config_path = Input(Types.String, required=True)
    streams_metadata_path = Input(Types.String, required=True)
    validation_data_ratio = Input(Types.Float, default=DEFAULT_VALIDATION_DATA_RATIO)

    rearrange_data_task = rearrange_data(
        training_validation_config_path=training_validation_config_path,
        streams_metadata_path=streams_metadata_path,
        validation_data_ratio=validation_data_ratio,
    )



    # ------------------------------------------------------------

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