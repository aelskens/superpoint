import pickle

import numpy as np
import pytest
import torch

from src.superpoint import (
    FlexibleSuperPoint,
    KeypointDescriptor,
    KeypointDetector,
    VGGLikeEncoder,
    numpy_image_to_torch,
    remove_batch_dimension,
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


@pytest.mark.parametrize(
    "img, expected",
    [(wsi, wsi_expected), (building, building_expected)],
)
def test_cohrence_with_superpoint(img, expected) -> None:
    """Unit test to validate that the results of FlexibleSuperPoint correspond to that of the original implementation."""

    def _compare(dict1, dict2):
        for k in dict1:
            if not torch.all(dict1[k] == dict2[k]):
                return False

        return True

    model = FlexibleSuperPoint(
        encoder=VGGLikeEncoder(), detector=KeypointDetector(max_num_keypoints=20000), descriptor=KeypointDescriptor()
    ).eval()
    model.load("https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth", agglomerated=True)

    assert _compare(model.extract(numpy_image_to_torch(img), resize=None), expected)


@pytest.mark.parametrize(
    "img, mode, expected",
    [(wsi, "complete", wsi_expected), (wsi, "split", wsi_expected)],
)
def test_opencv_interface(img, mode, expected) -> None:
    """Unit test to validate that the OPENCV interface yields the same results as the original implementation."""

    model = FlexibleSuperPoint(
        encoder=VGGLikeEncoder(), detector=KeypointDetector(max_num_keypoints=20000), descriptor=KeypointDescriptor()
    ).eval()
    model.load("https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth", agglomerated=True)

    match mode:
        case "complete":
            _, desc = model.detectAndCompute(img)
        case "split":
            kp = model.detect(img)
            _, desc = model.compute(img, kp)

    assert np.all(remove_batch_dimension(expected["descriptors"]).numpy() == desc)
