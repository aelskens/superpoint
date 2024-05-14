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

from .utils import numpy_image_to_torch, remove_batch_dimension
