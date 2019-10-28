from flytekit.sdk.types import Types
from flytekit.sdk.tasks import python_task, dynamic_task, outputs, inputs
from flytekit.sdk.workflow import workflow_class, Output, Input
from utils.frame_sampling.luminance_sampling import luminance_sample_collection


S3_PREFIX = "s3://lyft-modelbuilder/metadata"
SESSION_PATH_FORMAT = "{s3_prefix}/data/collections/{session_id}/{sub_path}/{stream_name}".format(s3_prefix=S3_PREFIX)
DEFAULT_SESSION_ID = (
    "1538521877,1538521964"
)
DEFAULT_INPUT_SUB_PATH = "raw"
DEFAULT_INPUT_STREAM = "cam-rgb-1.avi"


@inputs(input_blobs=[Types.Blob])
@inputs(file_names=[Types.String])
@inputs(session_id=Types.String)
@inputs(out_sub_path=Types.String)
@inputs(out_stream=Types.String)
@outputs(success=Types.Boolean)  # just a placeholder really
@dynamic_task(version="1")
def copy_blobs_to_collection_output(
    wf_params, input_blobs, file_names, session_id, out_sub_path, out_stream, success
):
    if out_sub_path == "raw":
        raise Exception(
            "Dont Ingest to the `raw` subpath. Data in `raw` is to remain un-changed"
        )

    s3_target_session_path = SESSION_PATH_FORMAT.format(
        session_id=session_id, sub_path=out_sub_path, stream_name=out_stream
    )

    for i in range(0, len(input_blobs)):
        location = s3_target_session_path + "/" + file_names[i]

        out_blob = Types.Blob.create_at_known_location(location)
        with out_blob as out_writer:
            with input_blobs[i] as in_reader:
                out_writer.write(in_reader.read())
        # out_blob.set(input_blobs[i])

    # write metadata:
    location = s3_target_session_path + "/.generated_by"
    out_blob = Types.Blob.create_at_known_location(location)
    with out_blob as out_writer:
        out_writer.write((f"workflow_id: {wf_params.execution_id}").encode("utf-8"))

    success.set(True)


@inputs(

)
@outputs(

)
@python_task
def copy_blobs_to_collection_output():
    pass


@inputs(
    session_ids=[Types.Integer],
    input_sub_path=Types.String,
    input_stream=Types.String,
)
@outputs(

)
@python_task
def extract_from_video_collections():
    pass


@inputs(

)
@outputs(

)
@dynamic_task
def luminance_select_collections


@workflow_class
class DataPreparationWorkflow:
    session_ids = Input(Types.String, required=True, default=DEFAULT_SESSION_ID, help="IDs of video sessions to extract frames out of")
    input_sub_path = Input(Types.String, default=DEFAULT_INPUT_SUB_PATH)
    input_stream = Input(Types.String, default=DEFAULT_INPUT_STREAM)
    output_sub_path = Input(Types.String, default=DEFAULT_OUTPUT_SUBPATH)
    output_stream = Input(Types.String, default=DEFAULT_OUTPUT_STREAM)

    extract_from_video_collection_task = extract_from_video_collections(
        session_ids=session_ids,
        input_sub_path=input_sub_path,
        input_stream=input_stream,
    )


    luminance_select_collections_task = luminance_select_collections(
        session_ids_string=session_ids,
        input_sub_path=input_sub_path,
        input_stream=input_stream,
        n_clusters=n_clusters,
        sample_size=sample_size,
        random_seed=random_seed,
    )


@workflow_class
class VideoToRawFrames:
    session_ids = Input(Types.String, required=True, default=DEFAULT_SESSION_ID, help="IDs of video sessions to extract frames out of")
    input_sub_path = Input(Types.String, default=DEFAULT_INPUT_SUB_PATH)
    input_stream = Input(Types.String, default=DEFAULT_INPUT_STREAM)
    output_sub_path = Input(Types.String, default=DEFAULT_OUTPUT_SUBPATH)
    output_stream = Input(Types.String, default=DEFAULT_OUTPUT_STREAM)

    extract_from_video_collection_task = extract_from_video_collections(
        session_ids=session_ids,
        input_sub_path=input_sub_path,
        input_stream=input_stream,
    )

    copy_blobs_to_collection_output_task = copy_blobs_to_collection_output(
        input_blobs=extract_from_video_collection_task.outputs.image_blobs,
        file_names=extract_from_video_collection_task.outputs.file_names,
        session_ids=session_ids,
        out_sub_path=output_sub_path,
        out_stream=output_stream,
    )

    create_blob_archives_task = create_blob_archives(
        input_zip_blobs=create_blob_archives_task.outputs.out_zip_blobs,
        session_ids_strings=session_ids,
        out_sub_path=output_sub_path,
        out_stream=output_stream,
    )


@workflow_class
class LuminanceSampleCollection:
    session_ids = Input(Types.String, default=TEST_SESSION_IDS)
    input_sub_path = Input(Types.String, default=DEFAULT_SELECTION_SUB_PATH)
    input_stream = Input(Types.String, default=DEFAULT_SELECTION_STREAM)
    n_clusters = Input(Types.Integer, default=DEFAULT_LUMINANCE_N_CLUSTERS)
    sample_size = Input(Types.Integer, default=DEFAULT_LUMINANCE_SAMPLE_SIZE)
    output_sub_path = Input(Types.String, default=DEFAULT_SELECTION_OUTPUT_SUB_PATH)
    output_stream = Input(Types.String, default=DEFAULT_SELECTION_STREAM)
    random_seed = Input(Types.Integer, default=DEFAULT_RANDOM_SEED)

    luminance_select_collections_task = luminance_select_collections(
        session_ids_string=session_ids,
        input_sub_path=input_sub_path,
        input_stream=input_stream,
        n_clusters=n_clusters,
        sample_size=sample_size,
        random_seed=random_seed,
    )

