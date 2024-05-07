from typing import Callable, Optional, Tuple

import kornia
import numpy as np
import torch
from torchvision import transforms


def numpy_image_to_torch(img: np.ndarray) -> torch.Tensor:
    """Convert numpy image to torch.

    :param img: The input numpy image to convert to torch.
    :type img: np.ndarray
    :return: The converted torch image.
    :rtype: torch.Tensor
    """

    numpy_to_torch = transforms.ToTensor()

    if img.dtype == float and img.dtype != np.float32:
        img = img.astype(np.float32)

    return numpy_to_torch(img).unsqueeze(0)


def smart_loader(path: str) -> Callable:
    """Determine which state_dict loader to use.

    :param path: The path to the model's .pth file.
    :type path: str
    :return: The appropriate loader to use.
    :rtype: Callable
    """

    if "http" in path:
        return torch.hub.load_state_dict_from_url

    return torch.load


class ImagePreprocessor:
    def __init__(
        self,
        resize: Optional[int] = None,
        side: str = "long",
        align_corners: Optional[bool] = None,
        antialias: bool = True,
    ) -> None:
        super().__init__()
        self.resize = resize
        self.side = side
        self.align_corners = align_corners
        self.antialias = antialias

    def __call__(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resize and preprocess an image, return image and resize scale"""
        h, w = img.shape[-2:]
        if self.resize is not None:
            img = kornia.geometry.transform.resize(
                img,
                self.resize,
                side=self.side,
                antialias=self.antialias,
                align_corners=self.align_corners,
            )
        scale = torch.Tensor([img.shape[-1] / w, img.shape[-2] / h]).to(img)
        return img, scale


def simple_nms(scores, nms_radius: int):
    """Fast Non-maximum suppression to remove nearby points"""
    assert nms_radius >= 0

    def max_pool(x):
        return torch.nn.functional.max_pool2d(x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def top_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0, sorted=True)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations"""
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor(
        [(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
    ).to(
        keypoints
    )[None]
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    align_corners = True if torch.__version__ >= "1.3" else None
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", align_corners=align_corners
    )
    descriptors = torch.nn.functional.normalize(descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


def remove_batch_dimension(t: torch.Tensor) -> torch.Tensor:
    return t.squeeze(0)
