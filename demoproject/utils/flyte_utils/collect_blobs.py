from os.path import join, isfile, basename
from os import listdir
from flytekit.sdk.types import Types

def collect_blobs(folder_path):
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