from flytekit.sdk.types import Types
from flytekit.sdk.tasks import python_task, dynamic_task, inputs, outputs

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
@python_task(cache=True, cache_version="1")
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
    """
    # Get the latest execution of the data_prep_workflow
    latest_dataprep_wf_execution = fetch_workflow_latest_execution(
        service_name=SERVICE_NAME,
        workflow_name=DATAPREP_WORKFLOW_NAME,
        service_instance=DEFAULT_SERVICE_INSTANCE,
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
    """
    a = Types.MultiPartBlob()
    training_clean_mpblob.set(a)
    training_dirty_mpblob.set(a)
    validation_clean_mpblob.set(a)
    validation_dirty_mpblob.set(a)