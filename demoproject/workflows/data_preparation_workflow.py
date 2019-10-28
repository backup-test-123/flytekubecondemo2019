import zipfile

from flytekit.sdk.types import Types
from flytekit.sdk.tasks import python_task, dynamic_task, outputs, inputs
from flytekit.sdk.workflow import workflow_class, Output, Input
from demoproject.utils.frame_sampling.luminance_sampling import luminance_sample_collection


S3_PREFIX = "s3://lyft-modelbuilder/metadata/kubecon"
SESSION_PATH_FORMAT = "{s3_prefix}/data/collections/{session_id}/{sub_path}/{stream_name}".format(s3_prefix=S3_PREFIX)
DEFAULT_SESSION_ID = (
    "1538521877,1538521964"
)
DEFAULT_INPUT_SUB_PATH = "raw"
DEFAULT_INPUT_STREAM = "cam-rgb-1.avi"


@inputs(
    input_blobs=[Types.Blob],
    file_names=[Types.String],
    session_id=Types.String,
    out_sub_path=Types.String,
    out_stream=Types.String,
)
@outputs(success=Types.Boolean)  # just a placeholder really
@dynamic_task(version="1")
def copy_blobs_to_collection_output(
    wf_params, input_blobs, file_names, session_id, out_sub_path, out_stream, success
):
    if out_sub_path == "raw":
        raise Exception(
            "Don't Ingest to the `raw` subpath. Data in `raw` is to remain un-changed"
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


@inputs()
@outputs()
@python_task
def copy_blobs_to_collection_output():
    pass



@inputs(
    input_zip_blob=Types.Blob,
    n_clusters=Types.Integer,
    sample_size=Types.Integer,
    random_seed=Types.Integer,
)
@outputs(
    selected_image_blobs=[Types.Blob],
    selected_file_names=[Types.String]
)
@python_task(cache_version="1")
def luminance_select_collection_worker(
    wf_params,
    input_zip_blob,
    n_clusters,
    sample_size,
    random_seed,
    selected_image_blobs,
    selected_file_names,
):
    local_zip_file = wf_params.working_directory.get_named_tempfile("input_zip.zip")
    local_img_folder = wf_params.working_directory.get_named_tempfile("input_images")
    local_output_folder = wf_params.working_directory.get_named_tempfile(
        "output_images"
    )
    Types.Blob.fetch(input_zip_blob, local_zip_file)

    # extract
    with zipfile.ZipFile(local_zip_file) as zf:
        zf.extractall(local_img_folder)

    luminance_sample_collection(
        local_img_folder,
        local_output_folder,
        n_clusters=n_clusters,
        sample_size=sample_size,
        logger=wf_params.logging,
        random_seed=random_seed,
    )

    # FINISH
    blobs, files_names_list = blobs_from_folder_recursive(local_output_folder)
    selected_image_blobs.set(blobs)
    selected_file_names.set(files_names_list)






@inputs(session_ids_string=Types.String)
@inputs(input_sub_path=Types.String)
@inputs(input_stream=Types.String)
@inputs(n_clusters=Types.Integer)
@inputs(sample_size=Types.Integer)
@inputs(random_seed=Types.Integer)
@outputs(selected_image_blobs=[[Types.Blob]])
@outputs(selected_file_names=[[Types.String]])
@dynamic_task(cache_version="1", discoverable=True)
def luminance_select_collections(
    wf_params,
    session_ids_string,
    input_sub_path,
    input_stream,
    n_clusters,
    sample_size,
    random_seed,
    selected_image_blobs,
    selected_file_names,
):
    """
    This is a driver task that kicks off the `luminance_select_collection_worker`s. It assumes that session_ids
    is a comma separated list of session_ids. It will then execute `luminance_select_collection_worker` for
    each of those, with the same sub_paths and stream. random_seed will be applied to each sub_task
    """
    all_session_ids = session_ids_string.split(",")
    if "_zip" in input_sub_path:
        input_stream = input_stream + ".zip"

    sub_tasks = [
        luminance_select_collection_worker(
            input_zip_blob=SESSION_PATH_FORMAT.format(
                session_id=session_id, sub_path=input_sub_path, stream_name=input_stream
            ),
            n_clusters=n_clusters,
            sample_size=sample_size,
            random_seed=random_seed,
        )
        for session_id in all_session_ids
    ]
    selected_image_blobs.set(
        [sub_task.outputs.selected_image_blobs for sub_task in sub_tasks]
    )
    selected_file_names.set(
        [sub_task.outputs.selected_file_names for sub_task in sub_tasks]
    )


@inputs(session_id=Types.String)
@inputs(input_sub_path=Types.String)
@inputs(input_stream=Types.String)
@outputs(image_blobs=[Types.Blob])
@outputs(file_names=[Types.String])
@python_task(cache_version="1", memory_hint=8000)
def extract_from_video_collection_worker(
    wf_params, session_id, input_sub_path, input_stream, image_blobs, file_names
):
    tmp_folder = wf_params.working_directory.get_named_tempfile("output")
    avi_local = wf_params.working_directory.get_named_tempfile("input.avi")

    avi_blob_location = SESSION_PATH_FORMAT.format(
        session_id=session_id, sub_path=input_sub_path, stream_name=input_stream
    )
    Types.Blob.fetch(avi_blob_location, avi_local)
    video_to_frames(avi_local, tmp_folder, False)

    blobs, files_names_list = blobs_from_folder_recursive(tmp_folder)
    image_blobs.set(blobs)
    file_names.set(files_names_list)


@inputs(session_ids=Types.String)
@inputs(input_sub_path=Types.String)
@inputs(input_stream=Types.String)
@outputs(image_blobs=[[Types.Blob]])
@outputs(file_names=[[Types.String]])
@dynamic_task(cache_version="1")  # TODO: memory_hint=8000, discoverable=True)
def extract_from_video_collections(
    wf_params, session_ids, input_sub_path, input_stream, image_blobs, file_names
):
    """
    This is a driver task that kicks off the `extract_from_avi_collection_worker`s. It assumes that session_ids
    is a comma separated list of session_ids. It will then execute extract_from_avi_collection for
    each of those, with the same sub_paths and stream
    """
    all_session_ids = session_ids.split(",")

    sub_tasks = [
        extract_from_video_collection_worker(
            session_id=session_id,
            input_sub_path=input_sub_path,
            input_stream=input_stream,
        )
        for session_id in all_session_ids
    ]

    image_blobs.set([sub_tasks.outputs.image_blobs for sub_tasks in sub_tasks])
    file_names.set([sub_tasks.outputs.file_names for sub_tasks in sub_tasks])



@workflow_class
class DataPreparationWorkflow:
    session_ids = Input(Types.String, required=True, help="IDs of video sessions to extract frames out of")
    video_sessions_path_prefix = Input(Types.String, required=True, help="")
    input_sub_path = Input(Types.String, default=DEFAULT_INPUT_SUB_PATH)
    input_stream = Input(Types.String, default=DEFAULT_INPUT_STREAM)
    sampling_random_seed = Input(Types.Integer, default=DEFAULT_RANDOM_SEED)
    sampling_n_clusters = Input(Types.Integer, default=DEFAULT_LUMINANCE_N_CLUSTERS)
    sampling_sample_size = Input(Types.Integer, default=DEFAULT_LUMINANCE_SAMPLE_SIZE)
    output_sub_path = Input(Types.String, default=DEFAULT_OUTPUT_SUBPATH)
    output_stream = Input(Types.String, default=DEFAULT_OUTPUT_STREAM)


    extract_from_video_collection_task = extract_from_video_collections(
        session_ids=session_ids,
        video_sessions_path_prefix=video_sessions_path_prefix,
        input_sub_path=input_sub_path,
        input_stream=input_stream,
    )

    luminance_select_collections_task = luminance_select_collections(
        session_ids_string=session_ids,
        input_sub_path=input_sub_path,
        input_stream=input_stream,
        n_clusters=sampling_n_clusters,
        sample_size=sampling_sample_size,
        random_seed=sampling_random_seed,
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

