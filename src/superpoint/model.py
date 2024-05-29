# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

# Adapted by Remi Pautrat, Philipp Lindenberger
# Adapted by Arthur Elskens

import os
from pathlib import Path
from shutil import ExecError
from typing import Any, Optional, Sequence, overload

import cv2
import numpy as np
import torch
from kornia.color import rgb_to_grayscale
from torch import nn

from .utils import (
    ArrayLike,
    ImagePreprocessor,
    MatLike,
    image_to_tensor,
    keypoint_to_tensor,
    remove_batch_dimension,
    sample_descriptors,
    simple_nms,
    smart_loader,
    tensor_to_cv_keypoint,
    top_k_keypoints,
)

MODELS_PATH = os.path.join(Path(os.path.abspath(__file__)).parents[2], "models")


class VGGLikeEncoder(nn.Module):
    """VGG-like encoder."""

    def __init__(self, activation_func: nn.Module = nn.ReLU(inplace=True)) -> None:
        """Constructor of VGGLikeEncoder class.

        :param activation_func: The activation function to use, defaults to nn.ReLU(inplace=True).
        :type activation_func: nn.Module, optional
        """

        super().__init__()

        self.activation = activation_func
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        c1, c2, c3, c4 = 64, 64, 128, 128

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Encode the image.

        :param img: The image to encode.
        :type img: torch.Tensor
        :return: The encoded image.
        :rtype: torch.Tensor
        """

        assert isinstance(img, torch.Tensor), "The given input image is not a Tensor."

        if img.shape[1] == 3:
            img = rgb_to_grayscale(img)

        x = self.activation(self.conv1a(img))
        x = self.activation(self.conv1b(x))
        x = self.pool(x)
        x = self.activation(self.conv2a(x))
        x = self.activation(self.conv2b(x))
        x = self.pool(x)
        x = self.activation(self.conv3a(x))
        x = self.activation(self.conv3b(x))
        x = self.pool(x)
        x = self.activation(self.conv4a(x))
        x = self.activation(self.conv4b(x))

        return x

    @overload
    def load(self, p: str) -> None: ...

    @overload
    def load(self, p: dict) -> None: ...

    def load(self, p):
        """Load the model's weights.

        :param p: The path to or the dictionary of weights to load into the model.
        :type p: str | dict
        """

        if isinstance(p, str):
            loader = smart_loader(p)
            p = loader(p)

        assert isinstance(p, dict)
        self.load_state_dict(p)

    def save(self, outpath: Optional[str] = None) -> None:
        """Save the model's weights.

        :param outpath: Where to save the weights. If None, use the model subdirectory,
        defaults to None.
        :type outpath: Optional[str], optional
        """

        if not outpath:
            outpath = f"{MODELS_PATH}/{self.__class__.__name__}.pth"

        torch.save(self.state_dict(), outpath)


class KeypointDetector(nn.Module):
    """Keypoint Detector Decoder."""

    def __init__(
        self,
        activation_func: nn.Module = nn.ReLU(inplace=True),
        max_num_keypoints: Optional[int] = None,
        nms_radius: int = 4,
        detection_threshold: float = 0.0005,
        remove_borders: int = 4,
    ) -> None:
        """Constructor for KeypointDetector class.

        :param activation_func: The activation function to use, defaults to nn.ReLU(inplace=True).
        :type activation_func: nn.Module, optional
        :param max_num_keypoints: The maximum number of keypoints to detect. If None, there are
        no limits, defaults to None.
        :type max_num_keypoints: Optional[int], optional
        :param nms_radius: The radius of the kernel used for the NMS step, defaults to 4.
        :type nms_radius: int, optional
        :param detection_threshold: The score threshold under which the keypoints are discarded,
        defaults to 0.0005.
        :type detection_threshold: float, optional
        :param remove_borders: The distance to the borders under which the keypoints are
        discarded, defaults to 4
        :type remove_borders: int, optional
        """

        super().__init__()
        self.max_num_keypoints = max_num_keypoints
        self.nms_radius = nms_radius
        self.detection_threshold = detection_threshold
        self.remove_borders = remove_borders

        self.activation = activation_func

        c4, c5 = 128, 256

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Forward pass of KeypointDetector.

        :param x: The input of the detector network.
        :type x: torch.Tensor
        :return: The keypoints and their scores.
        :rtype: tuple[list[torch.Tensor], torch.Tensor]
        """

        assert isinstance(x, torch.Tensor), "The given input is not a Tensor."

        # Compute the dense keypoint scores
        cPa = self.activation(self.convPa(x))
        scores = self.convPb(cPa)
        scores = nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        scores = simple_nms(scores, self.nms_radius)

        # Discard keypoints near the image borders
        if self.remove_borders:
            pad = self.remove_borders
            scores[:, :pad] = -1
            scores[:, :, :pad] = -1
            scores[:, -pad:] = -1
            scores[:, :, -pad:] = -1

        # Extract keypoints
        best_kp = torch.where(scores > self.detection_threshold)
        scores = scores[best_kp]

        # Separate into batches
        keypoints = [torch.stack(best_kp[1:3], dim=-1)[best_kp[0] == i] for i in range(b)]
        scores = [scores[best_kp[0] == i] for i in range(b)]

        # Keep the k keypoints with highest score
        if self.max_num_keypoints is not None:
            keypoints, scores = list(
                zip(*[top_k_keypoints(k, s, self.max_num_keypoints) for k, s in zip(keypoints, scores)])
            )

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        return keypoints, torch.stack(scores, 0)

    @overload
    def load(self, p: str) -> None: ...

    @overload
    def load(self, p: dict) -> None: ...

    def load(self, p):
        """Load the model's weights.

        :param p: The path to or the dictionary of weights to load into the model.
        :type p: str | dict
        """

        if isinstance(p, str):
            loader = smart_loader(p)
            p = loader(p)

        assert isinstance(p, dict)
        self.load_state_dict(p)

    def save(self, outpath: Optional[str] = None) -> None:
        """Save the model's weights.

        :param outpath: Where to save the weights. If None, use the model subdirectory,
        defaults to None.
        :type outpath: Optional[str], optional
        """

        if not outpath:
            outpath = f"{MODELS_PATH}/{self.__class__.__name__}.pth"

        torch.save(self.state_dict(), outpath)


class KeypointDescriptor(nn.Module):
    """Keypoint Descriptor Decoder."""

    def __init__(self, activation_func: nn.Module = nn.ReLU(inplace=True), descriptor_dim: int = 256) -> None:
        """Constructor of KeypointDescriptor class.

        :param activation_func: The activation function to use, defaults to nn.ReLU(inplace=True).
        :type activation_func: nn.Module, optional
        :param descriptor_dim: The dimensions of the descriptors, defaults to 256.
        :type descriptor_dim: int, optional
        """

        super().__init__()
        self.activation = activation_func

        c4, c5 = 128, 256

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, descriptor_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, keypoints: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass of KeypointDescriptor.

        :param x: The input of the descriptor network.
        :type x: torch.Tensor
        :param keypoints: The keypoints to describe.
        :type keypoints: list[torch.Tensor]
        :return: The descriptors for each keypoints.
        :rtype: torch.Tensor
        """

        assert isinstance(x, torch.Tensor), "The given input, x, is not a Tensor."

        # Compute the dense descriptors
        cDa = self.activation(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        # Extract descriptors
        descriptors = [sample_descriptors(k[None], d[None], 8)[0] for k, d in zip(keypoints, descriptors)]

        return torch.stack(descriptors, 0).transpose(-1, -2).contiguous()

    @overload
    def load(self, p: str) -> None: ...

    @overload
    def load(self, p: dict) -> None: ...

    def load(self, p):
        """Load the model's weights.

        :param p: The path to or the dictionary of weights to load into the model.
        :type p: str | dict
        """

        if isinstance(p, str):
            loader = smart_loader(p)
            p = loader(p)

        assert isinstance(p, dict)
        self.load_state_dict(p)

    def save(self, outpath: Optional[str] = None) -> None:
        """Save the model's weights.

        :param outpath: Where to save the weights. If None, use the model subdirectory,
        defaults to None.
        :type outpath: Optional[str], optional
        """

        if not outpath:
            outpath = f"{MODELS_PATH}/{self.__class__.__name__}.pth"

        torch.save(self.state_dict(), outpath)


class FlexibleSuperPoint(nn.Module):
    """FlexibleSuperPoint Convolutional Detector and/or Descriptor."""

    def __init__(
        self, encoder: nn.Module, detector: Optional[nn.Module] = None, descriptor: Optional[nn.Module] = None
    ) -> None:
        """Constructor of FlexibleSuperPoint class.

        :param encoder: The Encoder block.
        :type encoder: nn.Module
        :param detector: The detection head, if None then the detection step is skipped. It defaults to None.
        :type detector: Optional[nn.Module], optional
        :param descriptor: The description head, if None then the description step is skipped. It defaults to
        None.
        :type descriptor: Optional[nn.Module], optional
        """

        super().__init__()

        self.encoder = encoder
        self.detector = detector
        self.descriptor = descriptor

    def forward(
        self, img: torch.Tensor, keypoints: Optional[list[torch.Tensor]] = None
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Detect and describe keypoints from the input image.

        :param img:  The image from which the keypoints will be described and described.
        :type img: torch.Tensor
        :param keypoints: The keypoints to describe, defaults to None.
        :type keypoints: Optional[list[torch.Tensor]], optional
        :return: The keypoints along with their scores and descriptors.
        :rtype: tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]
        """

        assert isinstance(img, torch.Tensor), "The given input image is not a Tensor."

        scores, descriptors = None, None

        if img.shape[1] == 3:
            img = rgb_to_grayscale(img)

        feat = self.encoder(img)

        if self.detector:
            keypoints, scores = self.detector(feat)

        if self.descriptor:
            assert keypoints is not None
            descriptors = self.descriptor(feat, keypoints)

        return torch.stack(keypoints, 0) if keypoints else None, scores, descriptors

    def load(self, path: str = MODELS_PATH, agglomerated: bool = False) -> None:
        """Load the model's weights.

        :param path: The path to the weights to load into the model, defaults to MODELS_PATH.
        :type path: str, optional
        :param agglomerated: Whether the weights to load are grouped in a single file or not,
        defaults to False.
        :type agglomerated: bool, optional
        """

        encoder_params, detector_params, descriptor_params = {}, {}, {}

        if agglomerated:
            loader = smart_loader(path)
            model = loader(path)

            for k, v in model.items():
                if self.encoder and k.split(".")[0] in (module[0] for module in self.encoder.named_modules()):
                    encoder_params[k] = v
                elif self.detector and k.split(".")[0] in (module[0] for module in self.detector.named_modules()):
                    detector_params[k] = v
                elif self.descriptor and k.split(".")[0] in (module[0] for module in self.descriptor.named_modules()):
                    descriptor_params[k] = v

        for block, params in [
            (b, p)
            for b, p in zip(
                (self.encoder, self.detector, self.descriptor), (encoder_params, detector_params, descriptor_params)
            )
            if b is not None
        ]:
            block.load(f"{path}/{block.__class__.__name__}.pth" if not params else params)

    @torch.inference_mode()
    def detect(self, img: MatLike) -> Sequence[cv2.KeyPoint]:
        """Implementation of OpenCV's Feature2D `detect()` method.

        :param img: The image from which the keypoints will be detected.
        :type img: MatLike
        :return: The detected keypoints.
        :rtype: Sequence[cv2.KeyPoint]
        """

        if self.detector is None:
            raise ExecError("Cannot detect keypoints if the model has no Detection decoder.")

        tf_img = image_to_tensor(img)

        tmp_desc = self.descriptor
        self.descriptor = None

        keypoints, scores, _ = self.forward(tf_img)
        self.descriptor = tmp_desc

        assert keypoints is not None and scores is not None
        tmp = []
        for kp, score in zip(remove_batch_dimension(keypoints), remove_batch_dimension(scores)):
            tmp.append(cv2.KeyPoint(x=kp[0].item(), y=kp[1].item(), size=1.0, response=score.item()))

        return tuple(tmp)

    @torch.inference_mode()
    def compute(self, img: MatLike, keypoints: ArrayLike) -> tuple[Sequence[cv2.KeyPoint], np.ndarray]:
        """Implementation of OpenCV's Feature2D `detect()` method.

        :param img: The image from which the keypoints will be described.
        :type img: MatLike
        :param keypoints: The keypoints to describe.
        :type keypoints: ArrayLike
        :return keypoints: The detected keypoints.
        :rtype: Sequence[cv2.KeyPoint]
        :return descriptors: The associated descriptors.
        :rtype: np.ndarray
        """

        if self.descriptor is None:
            raise ExecError("Cannot describe keypoints if the model has no Descriptor decoder.")

        tf_img = image_to_tensor(img)
        tf_keypoints = [keypoint_to_tensor(keypoints)]

        tmp_detect = self.detector
        self.detector = None

        _, _, descriptors = self.forward(tf_img, tf_keypoints)
        self.detector = tmp_detect

        assert descriptors is not None

        return (
            keypoints if isinstance(keypoints, tuple) else tensor_to_cv_keypoint(tf_keypoints[0])
        ), remove_batch_dimension(descriptors).numpy(force=True)

    @torch.inference_mode()
    def detectAndCompute(self, img: MatLike) -> tuple[Sequence[cv2.KeyPoint], np.ndarray]:
        """Implementation of OpenCV's Feature2D `detect()` method.

        :param img: The image from which the keypoints will be described and described.
        :type img: MatLike
        :return keypoints: The detected keypoints.
        :rtype: Sequence[cv2.KeyPoint]
        :return descriptors: The associated descriptors.
        :rtype: np.ndarray
        """

        if self.detector is None or self.descriptor is None:
            raise ExecError(
                "Cannot detect and describe keypoints if the model has no Detector and/or no Descriptor decoders."
            )

        tf_img = image_to_tensor(img)
        _keypoints, scores, descriptors = self.forward(tf_img)

        assert _keypoints is not None and scores is not None and descriptors is not None
        keypoints = tensor_to_cv_keypoint(_keypoints, scores)

        return keypoints, remove_batch_dimension(descriptors).numpy(force=True)

    @torch.inference_mode()
    def extract(self, img: torch.Tensor, **conf: Any) -> dict[str, torch.Tensor]:
        """Perform extraction with online resizing.

        Note: implementation based on https://github.com/cvg/LightGlue/blob/main/lightglue/superpoint.py.

        :param img: The image from which the keypoints will be detected and described.
        :type img: torch.Tensor
        :return: The keypoints detected, their scores and their descriptors.
        :rtype: dict[str, torch.Tensor]
        """

        if img.dim() == 3:
            img = img[None]  # add batch dim

        assert img.dim() == 4 and img.shape[0] == 1

        img, scales = ImagePreprocessor(**{**{"resize": 1024}, **conf})(img)

        keypoints, scores, descriptors = self.forward(img, None)
        assert keypoints is not None and scores is not None and descriptors is not None

        keypoints = (keypoints + 0.5) / scales[None] - 0.5

        return {
            "keypoints": keypoints,
            "keypoint_scores": scores,
            "descriptors": descriptors,
        }
