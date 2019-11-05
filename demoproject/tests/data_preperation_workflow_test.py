from workflows.data_preparation_workflow import download_video_worker
import pytest
from flytekit.sdk.types import Types


def test_download_video_worker():
    res = download_video_worker.unit_test(
        # video_external_path='s3://my-dataset-s3-bucket/InCabin-demo/videos/1539391807_cam-rgb-1.avi'
        # video_external_path='http://localhost:9000/my-dataset-s3-bucket/InCabin-demo/videos/1537396662_cam-rgb-1.avi?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minio%2F20191104%2F%2Fs3%2Faws4_request&X-Amz-Date=20191104T223940Z&X-Amz-Expires=432000&X-Amz-SignedHeaders=host&X-Amz-Signature=fa675562beff8860ffebd9e7ed4912eccac14ce1f38011d64d22224b607977e8'
        video_external_path='s3://lyft-modelbuilder/metadata/_FlyteKubeconDemo2019Dataset/1537396662_cam-rgb-2.avi'
    )
    print(res)
    assert True