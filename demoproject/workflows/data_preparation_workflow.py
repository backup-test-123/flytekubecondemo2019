import zipfile
from os import listdir
from os.path import join, isfile, basename

from flytekit.common import utils as flytekit_utils
from flytekit.sdk.types import Types
from flytekit.sdk.tasks import python_task, dynamic_task, outputs, inputs
from flytekit.sdk.workflow import workflow_class, Output, Input
from demoproject.utils.frame_sampling.luminance_sampling import luminance_sample_collection
from demoproject.utils.video_tools.video_to_frames import video_to_frames


S3_PREFIX = "s3://lyft-modelbuilder/metadata/kubecon"
SESSION_PATH_FORMAT = "{s3_prefix}/data/collections/{session_id}/{sub_path}/{stream_name}".format(s3_prefix=S3_PREFIX)
DEFAULT_SESSION_ID = (
    "1538521877,1538521964"
)
DEFAULT_INPUT_SUB_PATH = "raw"
DEFAULT_INPUT_STREAM = "cam-rgb-1.avi"

DEFAULT_RANDOM_SEED = 0
DEFAULT_LUMINANCE_N_CLUSTERS = 4
DEFAULT_LUMINANCE_SAMPLE_SIZE = 2


def blobs_from_folder(folder_path):
    """
    Read all the files under a local path as blobs and return them, together with their
    base names (filenames without paths). Note that this function does not read blobs from
    sub folders

    :returns: (list of all the blobs, the corresponding file base names)
    """
    # Get the full paths of all the files, excluding sub-folders, under folder_path
    onlyfiles = [
        join(folder_path, f)
        for f in sorted(listdir(folder_path))
        if isfile(join(folder_path, f))
    ]
    my_blobs = []
    file_names = []
    for local_filepath in onlyfiles:

        my_blob = Types.Blob()
        with my_blob as fileobj:
            with open(local_filepath, mode="rb") as file:  # b is important -> binary
                fileobj.write(file.read())
        my_blobs.append(my_blob)
        file_names.append(basename(local_filepath))
    return my_blobs, file_names


def create_multipartblob_from_folder(folder_path):
    """
    """
    # Get the full paths of all the files, excluding sub-folders, under folder_path
    onlyfiles = [
        join(folder_path, f)
        for f in sorted(listdir(folder_path))
        if isfile(join(folder_path, f))
    ]
    mpblob = Types.MultiPartBlob()
    file_names = []

    for local_filepath in onlyfiles:
        file_basename = basename(local_filepath)
        with mpblob.create_part(file_basename) as fileobj:
            with open(local_filepath, mode='rb') as file:
                fileobj.wirte(file.read())
        file_names.append(file_basename)

    return mpblob, file_names


@inputs(
    raw_frames_multiblob=Types.MultiPartBlob,
    n_clusters=Types.Integer,
    sample_size=Types.Integer,
    random_seed=Types.Integer,
)
@outputs(
    selected_image_mpblob=Types.MultiPartBlob,
    selected_file_names=[Types.String]
)
@python_task(cache_version="1")
def luminance_select_collection_worker(
    wf_params,
    raw_frames_mpblob,
    n_clusters,
    sample_size,
    random_seed,
    selected_image_mpblob,
    selected_file_names,
):

    with flytekit_utils.AutoDeletingTempDir("output_images") as local_output_dir:
        raw_frames_mpblob.download()

        luminance_sample_collection(
            raw_frames_dir=raw_frames_mpblob.local_path,
            sampled_frames_out_dir=local_output_dir.local_path,
            n_clusters=n_clusters,
            sample_size=sample_size,
            logger=wf_params.logging,
            random_seed=random_seed,
        )

        mpblob, files_names_list = create_multipartblob_from_folder(local_output_dir.local_path)
        selected_image_mpblob.set(mpblob)
        selected_file_names.set(files_names_list)


@inputs(session_ids_string=Types.String)
@inputs(input_sub_path=Types.String)
@inputs(input_stream=Types.String)
@inputs(n_clusters=Types.Integer)
@inputs(sample_size=Types.Integer)
@inputs(random_seed=Types.Integer)
@outputs(selected_image_blobs=[[Types.Blob]])
@outputs(selected_file_names=[[Types.String]])
@dynamic_task(cache_version="1")
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


@inputs(
    session_id=Types.String,
    input_sub_path=Types.String,
    input_stream=Types.Strin
)
@outputs(
    image_blobs=[Types.Blob],
    file_names=[Types.String],
)
@python_task(cache_version="1", memory_request=8000)
def extract_from_video_collection_worker(
    wf_params, session_id, input_sub_path, input_stream, image_blobs, file_names
):
    tmp_folder = wf_params.working_directory.get_named_tempfile("output")
    video_local = wf_params.working_directory.get_named_tempfile("input.avi")

    video_blob_location = SESSION_PATH_FORMAT.format(
        session_id=session_id, sub_path=input_sub_path, stream_name=input_stream
    )
    Types.Blob.fetch(video_blob_location, video_local)
    video_to_frames(video_local, tmp_folder, False)

    blobs, files_names_list = blobs_from_folder(tmp_folder)
    image_blobs.set(blobs)
    file_names.set(files_names_list)


@inputs(
    session_ids=Types.String,
    input_sub_path=Types.String,
    input_stream=Types.String
)
@outputs(
    video_frames_multipartblobs=[[Types.Blob]],
    video_paths=[[Types.String]]
)
@dynamic_task(cache_version="1", memory_request=8000)
def extract_from_video_collections(
    wf_params, session_ids, input_sub_path, input_stream, videos_frames_multipartblobs, video_paths
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

    videos_frames_multipartblobs.set([sub_tasks.outputs.image_blobs for sub_tasks in sub_tasks])
    video_paths.set([sub_tasks.outputs.file_names for sub_tasks in sub_tasks])


@workflow_class
class DataPreparationWorkflow:
    session_ids = Input(Types.String, required=True, help="IDs of video sessions to extract frames out of")
    session_streams = Input(Types.String, required=True, help="Stream names of video sessions to extract frames out of")
    video_sessions_path_prefix = Input(Types.String, required=True,
                                       help="The path prefix where all the raw videos are stored")
    sampling_random_seed = Input(Types.Integer, default=DEFAULT_RANDOM_SEED)
    sampling_n_clusters = Input(Types.Integer, default=DEFAULT_LUMINANCE_N_CLUSTERS)
    sampling_sample_size = Input(Types.Integer, default=DEFAULT_LUMINANCE_SAMPLE_SIZE)

    extract_from_video_collection_task = extract_from_video_collections(
        session_ids=session_ids,
        video_sessions_path_prefix=video_sessions_path_prefix,
    )

    luminance_select_collections_task = luminance_select_collections(
        session_ids_string=session_ids,
        n_clusters=sampling_n_clusters,
        sample_size=sampling_sample_size,
        random_seed=sampling_random_seed,
    )

    raw_frames_multipartblobs = Output(extract_from_video_collection_task.outputs .outputs.video_frames_multipartblobs)
    raw_frames_metadata = Output(extract_from_video_collection_task.outputs.video_paths)

    sampled_frames_multipartblobs = Output(luminance_select_collections_task.outputs.sample_frames_multipartblobs)
    sampled_frames_metadata = Output(luminance_select_collections_task.outputs.video_paths)