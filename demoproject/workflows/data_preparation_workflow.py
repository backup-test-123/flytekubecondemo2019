import zipfile
from os import listdir
from os.path import join, isfile, basename

from flytekit.common import utils as flytekit_utils
from flytekit.sdk.types import Types
from flytekit.sdk.tasks import python_task, dynamic_task, outputs, inputs
from flytekit.sdk.workflow import workflow_class, Output, Input
from utils.frame_sampling.luminance_sampling import luminance_sample_collection
from utils.video_tools.video_to_frames import video_to_frames


STREAM_EXTERNAL_PATH_FORMAT = "{remote_prefix}/{stream_name}.{stream_extension}"

DEFAULT_RANDOM_SEED = 0
DEFAULT_LUMINANCE_N_CLUSTERS = 8
DEFAULT_LUMINANCE_SAMPLE_SIZE = 20


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
@python_task(cache=True, cache_version="1")
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
            sampled_frames_out_dir=local_output_dir.name,
            n_clusters=n_clusters,
            sample_size=sample_size,
            logger=wf_params.logging,
            random_seed=random_seed,
        )

        # Get the full paths of all the files, excluding sub-folders, under folder_path
        selected_file_names_in_folder = [
            f for f in sorted(listdir(local_output_dir.name))
            if isfile(join(local_output_dir.name, f))
        ]

        selected_image_mpblob.set(local_output_dir.name)
        selected_file_names.set(selected_file_names_in_folder)


@inputs(
    raw_frames_mpblobs=[Types.MultiPartBlob],
    n_clusters=Types.Integer,
    sample_size=Types.Integer,
    random_seed=Types.Integer
)
@outputs(
    selected_image_mpblobs=[Types.MultiPartBlob],
    selected_file_names=[[Types.String]],
)
@dynamic_task(cache=True, cache_version="1")
def luminance_select_collections(
    wf_params,
    raw_frames_mpblobs,
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
    # input_video_remote_path=Types.String,
    video_blob=Types.Blob,
)
@outputs(
    raw_frames_mpblob=Types.MultiPartBlob,
)
@python_task(cache=True, cache_version="1", memory_request='8000Mi')
def extract_from_video_collection_worker(
    wf_params, video_blob, raw_frames_mpblob,
):

    with flytekit_utils.AutoDeletingTempDir("output_images") as local_output_dir:

        # To keep the original basename visible. Optional
        video_local_path = join(local_output_dir.name, basename(video_blob.remote_location))
        video_blob.download(local_path=video_local_path)
        print("Video blob {} downloaded to {}".format(video_blob.remote_location, video_local_path))

        video_to_frames(
            video_filename=video_blob.local_path,
            output_dir=local_output_dir.name,
            skip_if_dir_exists=False
        )
        raw_frames_mpblob.set(local_output_dir.name)


@inputs(
    # video_remote_paths=[Types.String]
    video_blobs=[Types.Blob]
)
@outputs(
    raw_frames_mpblobs=[Types.MultiPartBlob],
)
@dynamic_task(cache=True, cache_version="1")
def extract_from_video_collections(
    wf_params, video_blobs, raw_frames_mpblobs,
):
    """
    This is a driver task that kicks off the `extract_from_avi_collection_worker`s. It assumes that session_ids
    is a comma separated list of session_ids. It will then execute extract_from_avi_collection for
    each of those, with the same sub_paths and stream
    """

    sub_tasks = [
        extract_from_video_collection_worker(
            video_blob=video_blob,
        )
        for video_blob in video_blobs
    ]

    raw_frames_mpblobs.set([sub_tasks.outputs.raw_frames_mpblob for sub_tasks in sub_tasks])


@inputs(
    video_external_path=Types.String,
)
@outputs(
    video_blob=Types.Blob,
)
@python_task(cache=True, cache_version='1')
def download_video_worker(
    wf_params, video_external_path, video_blob,
):
    # avi_local = wf_params.working_directory.get_named_tempfile("input.avi")
    with flytekit_utils.AutoDeletingTempDir("stream") as download_dir:
        local_path = join(download_dir.name, basename(video_external_path))
        b = Types.Blob.fetch(remote_path=video_external_path, local_path=local_path)
        video_blob.set(b)


@inputs(
    streams_external_storage_prefix=Types.String,
    streams_names=[Types.String],
    stream_extension=Types.String,
)
@outputs(
    downloaded_streams_blobs=[Types.Blob],
    downloaded_streams_names=[Types.String],
)
@dynamic_task(cache=True, cache_version='1', memory_request='800Mi')
def download_videos(
    wf_params, streams_external_storage_prefix, streams_names, stream_extension,
        downloaded_streams_blobs, downloaded_streams_names,
):
    blobs = []

    for stream_name in streams_names:
        stream_external_path = STREAM_EXTERNAL_PATH_FORMAT.format(
            remote_prefix=streams_external_storage_prefix,
            stream_name=stream_name,
            stream_extension=stream_extension,
        )
        download_task = download_video_worker(video_external_path=stream_external_path)
        yield download_task
        blobs.append(download_task.outputs.video_blob)

    downloaded_streams_blobs.set(blobs)
    downloaded_streams_names.set(streams_names)


@workflow_class
class DataPreparationWorkflow:
    streams_external_storage_prefix = Input(Types.String, required=True)
    streams_names = Input([Types.String], required=True)
    stream_extension = Input(Types.String, default="avi")

    # video_external_paths = Input([Types.String], required=True)
    sampling_random_seed = Input(Types.Integer, default=DEFAULT_RANDOM_SEED)
    sampling_n_clusters = Input(Types.Integer, default=DEFAULT_LUMINANCE_N_CLUSTERS)
    sampling_sample_size = Input(Types.Integer, default=DEFAULT_LUMINANCE_SAMPLE_SIZE)

    download_video_task = download_videos(
        streams_external_storage_prefix=streams_external_storage_prefix,
        streams_names=streams_names,
        stream_extension=stream_extension,
    )

    extract_from_video_collection_task = extract_from_video_collections(
        video_blobs=download_video_task.outputs.downloaded_streams_blobs,
    )

    luminance_select_collections_task = luminance_select_collections(
        raw_frames_mpblobs=extract_from_video_collection_task.outputs.raw_frames_mpblobs,
        n_clusters=sampling_n_clusters,
        sample_size=sampling_sample_size,
        random_seed=sampling_random_seed,
    )

    selected_frames_mpblobs = Output(luminance_select_collections_task.outputs.selected_image_mpblobs,
                                     sdk_type=[Types.MultiPartBlob])
    selected_frames_mpblobs_metadata = Output(luminance_select_collections_task.outputs.selected_file_names,
                                              sdk_type=[[Types.String]])
    streams_names_out = Output(streams_names, sdk_type=[Types.String])

