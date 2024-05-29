from typing import Callable, Optional, Sequence, Tuple, Union

import cv2
import kornia
import numpy as np
import torch
from torchvision import transforms

MatLike = Union[np.ndarray, torch.Tensor]
ArrayLike = Union[MatLike, Sequence[cv2.KeyPoint]]


def image_to_tensor(img: MatLike) -> torch.Tensor:
    """Convert image to torch.Tensor.

    :param img: The input image to convert to torch.Tensor.
    :type img: MatLike
    :return: The torch.Tensor image (dtype=torch.float32). Note that a batch dimension was added.
    :rtype: torch.Tensor
    """

    if isinstance(img, torch.Tensor):
        tmp = img.unsqueeze(0) if img.ndim < 4 else img
    elif isinstance(img, np.ndarray):
        tmp = numpy_image_to_torch(img)
    else:
        raise TypeError(f"The given input image type, {type(img)}, is not supported at the moment.")

    return tmp.type(torch.float32)


def numpy_image_to_torch(img: np.ndarray) -> torch.Tensor:
    """Convert numpy image to torch.Tensor.

    :param img: The input numpy image to convert to torch.
    :type img: np.ndarray
    :return: The torch.Tensor image.
    :rtype: torch.Tensor
    """

    numpy_to_torch = transforms.ToTensor()
    return numpy_to_torch(img).unsqueeze(0)


def keypoint_to_tensor(keypoints: ArrayLike) -> torch.Tensor:
    """Convert keypoints in the expected input of the descriptor head: torch.tensor(shape=(n_kp, 2)).

    :param keypoints: The keypoints (N) to convert as a tensor [N x 2].
    :type keypoints: ArrayLike
    :raises TypeError: Triggers if the input type is not supported.
    :return: The converted keypoints.
    :rtype: torch.Tensor
    """

    if isinstance(keypoints, tuple):
        return torch.stack([torch.tensor(kp.pt) for kp in keypoints])
    elif isinstance(keypoints, np.ndarray):
        return torch.tensor(keypoints)
    elif isinstance(keypoints, torch.Tensor):
        return keypoints
    else:
        raise TypeError(f"The input keypoints type, {type(keypoints)}, does not match the current implementation.")


def tensor_to_cv_keypoint(keypoints: torch.Tensor, scores: Optional[torch.Tensor] = None) -> Sequence[cv2.KeyPoint]:
    """Convert keypoints in the expected input of the descriptor head: torch.tensor(shape=(n_kp, 2)).

    :param keypoints: The keypoints (N) to convert as a tensor [N x 2].
    :type keypoints: ArrayLike
    :raises TypeError: Triggers if the input type is not supported.
    :return: The converted cv2 keypoints.
    :rtype: Sequence[cv2.KeyPoint]
    """

    if scores is None:
        scores = torch.ones(keypoints.shape[:-1])

    tmp = []
    for kp, score in zip(remove_batch_dimension(keypoints), remove_batch_dimension(scores)):
        tmp.append(cv2.KeyPoint(x=kp[0].item(), y=kp[1].item(), size=1.0, response=score.item()))

    return tuple(tmp)


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
    """Remove the batch dimension from Tensor.

    :param t: The tensor to adapt.
    :type t: torch.Tensor
    :return: The adapted tensor.
    :rtype: torch.Tensor
    """

    return t.squeeze(0)
