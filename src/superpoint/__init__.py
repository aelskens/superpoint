from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("superpoint")
except PackageNotFoundError:
    __version__ = "dev"

from .model import (
    FlexibleSuperPoint,
    KeypointDescriptor,
    KeypointDetector,
    VGGLikeEncoder,
)
from .utils import image_to_tensor, keypoint_to_tensor, remove_batch_dimension
