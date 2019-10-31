import zipfile
from os import listdir
from os.path import join, isfile, basename

from flytekit.common import utils as flytekit_utils
from flytekit.sdk.types import Types
from flytekit.sdk.tasks import python_task, dynamic_task, outputs, inputs
from flytekit.sdk.workflow import workflow_class, Output, Input
from utils.frame_sampling.luminance_sampling import luminance_sample_collection
from utils.video_tools.video_to_frames import video_to_frames


SESSION_PATH_FORMAT = "{remote_prefix}/data/collections/{session_id}/{sub_path}/{stream_name}"
# DEFAULT_SESSION_ID = (
#     "1538521877,1538521964"
# )
# DEFAULT_INPUT_SUB_PATH = "raw"
# DEFAULT_INPUT_STREAM = "cam-rgb-1.avi"

DEFAULT_RANDOM_SEED = 0
DEFAULT_LUMINANCE_N_CLUSTERS = 4
DEFAULT_LUMINANCE_SAMPLE_SIZE = 2


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
    raw_frames_mpblob=Types.MultiPartBlob,
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

        mpblob, selected_file_names_in_folder = create_multipartblob_from_folder(local_output_dir.local_path)
        selected_image_mpblob.set(mpblob)
        selected_file_names.set(selected_file_names_in_folder)


@inputs(
    raw_frames_mpblobs=[Types.MultiPartBlob],
    corresponding_videos_paths=[Types.String],
    n_clusters=Types.Integer,
    sample_size=Types.Integer,
    random_seed=Types.Integer
)
@outputs(
    selected_image_mpblobs=[Types.MultiPartBlob],
    selected_file_names=[[Types.String]],
)
@dynamic_task(cache_version="1")
def luminance_select_collections(
    wf_params,
    raw_frames_mpblobs,
    corresponding_videos_paths,
    n_clusters,
    sample_size,
    random_seed,
    selected_image_mpblobs,
    selected_file_names,
):
    """
    This is a driver task that kicks off the `luminance_select_collection_worker`s. It assumes that session_ids
    is a comma separated list of session_ids. It will then execute `luminance_select_collection_worker` for
    each of those, with the same sub_paths and stream. random_seed will be applied to each sub_task
    """

    sub_tasks = [
        luminance_select_collection_worker(
            raw_frames_mpblob=raw_frames_mpblob,
            n_clusters=n_clusters,
            sample_size=sample_size,
            random_seed=random_seed,
        )
        for raw_frames_mpblob in raw_frames_mpblobs
    ]
    selected_image_mpblobs.set(
        [sub_task.outputs.selected_image_mpblob for sub_task in sub_tasks]
    )
    selected_file_names.set(
        [sub_task.outputs.selected_file_names for sub_task in sub_tasks]
    )


@inputs(
    input_video_remote_path=Types.String,
)
@outputs(
    raw_frames_mpblob=Types.MultiPartBlob,
    video_file_name=Types.String,
)
@python_task(cache_version="1", memory_request='8000')
def extract_from_video_collection_worker(
    wf_params, input_video_remote_path, raw_frames_mpblob, video_file_name
):

    with wf_params.working_directory_get_named_tempfile("input.avi") as video_local_path:
        with flytekit_utils.AutoDeletingTempDir("output_images") as local_output_dir:
            Types.Blob.fetch(remote_path=input_video_remote_path, local_path=video_local_path)

            video_to_frames(
                video_filename=video_local_path,
                output_dir=local_output_dir,
                skip_if_dir_exists=False
            )

            mpblob, files_names_list = create_multipartblob_from_folder(local_output_dir.local_path)
            raw_frames_mpblob.set(mpblob)
            video_file_name.set(input_video_remote_path)


@inputs(
    video_remote_paths=[Types.String]
)
@outputs(
    raw_frames_mpblobs=[[Types.MultiPartBlob]],
    video_paths=[Types.String]
)
@dynamic_task(cache_version="1", memory_request='8000')
def extract_from_video_collections(
    wf_params, video_remote_paths, raw_frames_mpblobs, video_paths
):
    """
    This is a driver task that kicks off the `extract_from_avi_collection_worker`s. It assumes that session_ids
    is a comma separated list of session_ids. It will then execute extract_from_avi_collection for
    each of those, with the same sub_paths and stream
    """

    sub_tasks = [
        extract_from_video_collection_worker(
            input_video_remote_path=video_remote_path,
        )
        for video_remote_path in video_remote_paths
    ]

    raw_frames_mpblobs.set([sub_tasks.outputs.image_blobs for sub_tasks in sub_tasks])
    video_paths.set(t.outputs.video_file_name for t in sub_tasks)


@inputs(
    video_remote_prefix=Types.String,
    sub_path=Types.String,
    session_ids_str=Types.String,
    session_streams_str=Types.String,
)
@outputs(
    video_remote_paths=[Types.String],
)
@python_task(cache_version='1')
def generate_video_full_remote_paths(
    wf_params, video_remote_prefix, sub_path, session_ids_str, session_streams_str, video_remote_paths
):
    remote_paths = []

    session_ids = session_ids_str.split(',')
    session_streams = session_streams_str.split(',')

    video_path_info_pairs = zip(session_ids, session_streams)

    for session_id, session_stream in video_path_info_pairs:
        remote_paths.append(
            SESSION_PATH_FORMAT.format(
                remote_prefix=video_remote_prefix,
                session_id=session_id,
                sub_path=sub_path,
                stream_name=session_stream,
            )
        )
    video_remote_paths.set(remote_paths)


@workflow_class
class DataPreparationWorkflow:
    session_ids_str = Input(Types.String, required=True,
                        help="A string containing comma-separated IDs of video sessions to extract frames out of")
    session_streams_str = Input(Types.String, required=True,
                            help="A string containing comma-separated stream names of video sessions to extract frames out of")
    video_remote_prefix = Input(Types.String, required=True, help="The path prefix where all the raw videos are stored")
    sub_path = Input(Types.String, required=True,
                     help="The subpath where the streams are actually stored")
    sampling_random_seed = Input(Types.Integer, default=DEFAULT_RANDOM_SEED)
    sampling_n_clusters = Input(Types.Integer, default=DEFAULT_LUMINANCE_N_CLUSTERS)
    sampling_sample_size = Input(Types.Integer, default=DEFAULT_LUMINANCE_SAMPLE_SIZE)

    generate_video_full_remote_paths_task = generate_video_full_remote_paths(
        video_remote_prefix=video_remote_prefix,
        sub_path=sub_path,
        session_ids_str=session_ids_str,
        session_streams_str=session_streams_str,
    )

    extract_from_video_collection_task = extract_from_video_collections(
        video_remote_paths=generate_video_full_remote_paths_task.outputs.video_remote_paths
    )

    luminance_select_collections_task = luminance_select_collections(
        raw_frames_mpblobs=extract_from_video_collection_task.outputs.raw_frames_mpblobs,
        corresponding_videos_paths=extract_from_video_collection_task.outputs.video_paths,
        n_clusters=sampling_n_clusters,
        sample_size=sampling_sample_size,
        random_seed=sampling_random_seed,
    )

    selected_frames_mpblobs = Output(luminance_select_collections_task.outputs.selected_image_mpblobs,
                                     sdk_type=Types.MultiPartBlob)
    selected_frames_mpblobs_metadata = Output(luminance_select_collections_task.outputs.selected_file_names,
                                              sdk_type=[[Types.String]])
