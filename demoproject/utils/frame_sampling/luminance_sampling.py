import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

DEFAULT_IMAGE_EXTENSION = "png"
DEFAULT_N_TILES = (32, 32)


def luminance_sample_collection(
    raw_frames_dir,
    sampled_frames_out_dir,
    n_clusters,
    sample_size,
    logger,
    random_seed=0,
    n_tiles=DEFAULT_N_TILES,
):
    """
    Take all the frames from raw_images_dir, calculate their luminance vectors,
    and create n_clusters of those. Then copy sample_size amounts of images out
    of each n_clusters to the sampled_images_out_dir.
    """
    frame_ids, frame_lumin_vecs = analyze_image_collection(
        raw_frames_dir, n_tiles=n_tiles
    )

    logger.info("Starting to run K-means clustering...")
    kmeans = run_kmeans(
        data=frame_lumin_vecs, n_clusters=n_clusters, n_jobs=1, random_state=random_seed
    )

    logger.info("K-means clustering finished.")
    frame_group_df = pd.DataFrame({"frame_id": frame_ids, "label": kmeans.labels_})

    grouped = frame_group_df.groupby("label", as_index=False)

    np.random.seed(random_seed)
    sampled_groups = {}

    for name, group in grouped:
        sampled_df = group.loc[
            np.random.choice(
                group.index, min(sample_size, group.shape[0]), replace=False
            ),
            :,
        ]
        sampled_df.reset_index(inplace=True, drop=True)
        sampled_groups[name] = sampled_df

    sample_frames(
        sampled_groups,
        src_dir=raw_frames_dir,
        dest_dir=sampled_frames_out_dir,
        skip_if_dir_exists=False,
        logger=logger,
    )


def run_kmeans(data, n_clusters, n_jobs, random_state):
    kmeans = KMeans(
        init="k-means++",
        n_clusters=n_clusters,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    kmeans.fit(data)
    return kmeans


def sample_frames(sampled_groups, src_dir, dest_dir, skip_if_dir_exists, logger):
    dest_dir_path = Path(dest_dir)

    if skip_if_dir_exists and dest_dir_path.exists():
        logger.info(
            "Frame directory '{}' exists. " "Skipping frame sampling".format(dest_dir)
        )
        return

    if dest_dir_path.exists():
        shutil.rmtree(dest_dir)

    dest_dir_path = Path(dest_dir)
    dest_dir_path.mkdir(0o744, parents=True, exist_ok=True)
    for k in sorted(list(sampled_groups.keys())):
        logger.info(f"Group {k}")
        for _, row in sampled_groups[k].iterrows():
            basename = "{}.{}".format(row["frame_id"], DEFAULT_IMAGE_EXTENSION)
            src_filename = f"{src_dir}/{basename}"
            dest_filename = f"{dest_dir}/{basename}"
            logger.info(f"Selecting {src_filename} to -> {dest_filename}")
            shutil.copy(src_filename, dest_filename)


def analyze_image_collection(frames_directory, n_tiles):
    luminance_vectors = []
    frame_ids = []
    frame_filenames = sorted(
        f"{frames_directory}/{f}"
        for f in os.listdir(frames_directory)
        if f.endswith(DEFAULT_IMAGE_EXTENSION)
    )
    for frame_filename in frame_filenames:
        frame = cv2.imread(frame_filename, cv2.IMREAD_COLOR)
        frame_id = os.path.basename(frame_filename).split(".")[0]
        frame_ids.append(frame_id)
        luminance_vectors.append(calculate_frame_luminance_vector(frame, n_tiles))

    return frame_ids, luminance_vectors


def calculate_frame_luminance_vector(bgr_frame, n_tiles):
    """
    Takes a frame encoded in BGR color space and convert it to YCrCb color space to easily retrieve luminance information (Y channel).
    Then tile the frame's luminance channel. Then this function calculate the average luminance of all the tiles, and flatten them
    into a 1-D list of floating points

    :param bgr_frame:
    :param n_tiles:
    :return:
    """
    ycrcb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YCrCb)
    luminance_channel = ycrcb_frame[:, :, 0]
    tiles = tiling_one_frame(
        luminance_channel, n_vertical_tiles=n_tiles[0], n_horizontal_tiles=n_tiles[1]
    )

    def flatten(l):
        return [calculate_tile_avg_luminance(item) for sublist in l for item in sublist]

    return flatten(tiles)


def tiling_one_frame(frame, n_vertical_tiles, n_horizontal_tiles):
    """
    Divide a frame into (n_vertical_tiles*n_horizontal_tiles) tiles.

    :param frame:
    :param n_vertical_tiles:
    :param n_horizontal_tiles:
    :return:
    """
    height, width = frame.shape

    tile_height = height / n_vertical_tiles
    tile_width = width / n_horizontal_tiles

    tiles = np.empty([n_vertical_tiles, n_horizontal_tiles], dtype=object)

    for y in range(0, n_vertical_tiles):
        for x in range(0, n_horizontal_tiles):
            tiles[y][x] = frame[
                y*int(tile_height):(y+1)*int(tile_height),
                x*int(tile_width):(x+1)*int(tile_width),
            ]

    return tiles


def calculate_tile_avg_luminance(tile_y):
    """
    Takes a tile (of the luminance channel of the YCrCb image) and calculate the average luminance of this tile.

    :param tile_y:
    :return:
    """
    return np.mean(np.mean(tile_y, axis=0, dtype=np.float64), axis=0, dtype=np.float64)