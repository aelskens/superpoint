import pickle
import random
from time import perf_counter

import numpy as np
import pytest
import torch

from src.superpoint import (
    FlexibleSuperPoint,
    KeypointDescriptor,
    KeypointDetector,
    VGGLikeEncoder,
    numpy_image_to_torch,
)


def pickle_load(path):
    with open(path, "rb") as fp:
        out = pickle.load(fp)

    return out


# .npy of two different images: (i) a Whole-Slide Image and (ii) a building image
wsi = np.load("./tests/wsi.npy")
building = np.load("./tests/building.npy")

# The pickled output of LightGlue's SuperPoint implementation for both image
# with a `max_num_keypoints=20000`and `resize=None`
wsi_expected = pickle_load("./tests/superpoint_extract_wsi.pickle")
building_expected = pickle_load("./tests/superpoint_extract_building.pickle")


# REMARK: The following test fails because there are differences between the CPU and GPU results
# Indeed, there are small differences between the keypoint_scores on the different devices which induce
# a different ordering of the keypoints and descriptors
@pytest.mark.parametrize(
    "img, expected",
    [(wsi, wsi_expected), (building, building_expected)],
)
def _test_gpu(img, expected) -> None:
    def _compare(dict1, dict2):
        for k in dict1:
            if not torch.all(dict1[k] == dict2[k]):
                return False

        return True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA is not available. This test won't pass as long as CUDA is not available."

    model = (
        FlexibleSuperPoint(
            encoder=VGGLikeEncoder(),
            detector=KeypointDetector(max_num_keypoints=20000),
            descriptor=KeypointDescriptor(),
        )
        .eval()
        .to(device)
    )
    model.load("https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth", agglomerated=True)

    assert _compare(model.extract(numpy_image_to_torch(img).to(device), resize=None), expected)


@pytest.mark.parametrize(
    "img",
    [
        random.random() * np.ones_like(wsi) * wsi,
        random.random() * np.ones_like(building) * building,
    ],
)
def test_gpu_optimization(img) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA is not available. This test won't pass as long as CUDA is not available."

    model = (
        FlexibleSuperPoint(
            encoder=VGGLikeEncoder(),
            detector=KeypointDetector(max_num_keypoints=20000),
            descriptor=KeypointDescriptor(),
        )
        .eval()
        .to(device)
    )
    model.load("https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth", agglomerated=True)
    tensor_img = numpy_image_to_torch(img).to(device)
    model.extract(tensor_img, resize=None)

    # Time the second extraction as it takes time for the GPU warm itself
    start_time = perf_counter()
    model.extract(tensor_img, resize=None)
    end_time = perf_counter()
    gpu_run_time = end_time - start_time

    device = torch.device("cpu")

    model = (
        FlexibleSuperPoint(
            encoder=VGGLikeEncoder(),
            detector=KeypointDetector(max_num_keypoints=20000),
            descriptor=KeypointDescriptor(),
        )
        .eval()
        .to(device)
    )
    model.load("https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth", agglomerated=True)
    tensor_img = numpy_image_to_torch(img).to(device)

    start_time = perf_counter()
    model.extract(tensor_img, resize=None)
    end_time = perf_counter()
    cpu_run_time = end_time - start_time

    print(cpu_run_time, gpu_run_time)

    assert cpu_run_time > gpu_run_time
