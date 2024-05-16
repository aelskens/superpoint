# Superpoint

This repository implements a more flexible version of [SuperPoint](https://doi.org/10.1109/CVPRW.2018.00060) based on the lightglue library's [implementation](https://github.com/cvg/LightGlue/blob/main/lightglue/superpoint.py).

The main differences with lightglue's version are: (i) the model has been split into 3 blocks (encoder, detection head and description head), and (ii) the implementation of an OpenCV-like interface. Thanks to the splitting the model, one can use SuperPoint with more flexibility either by changing the architecture of one or more blocks or by using SuperPoint as a detector or as a descriptor.

## Installation

Install this repository using pip:
```bash
git clone https://github.com/aelskens/superpoint.git && cd superpoint
python -m pip install .
```

## Usage

Here is a minimal example:
```python
from superpoint import VGGLikeEncoder, KeypointDetector, KeypointDescriptor, FlexibleSuperPoint
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt

# Define each block
encoder = VGGLikeEncoder()
detector = KeypointDetector(max_num_keypoints=1024)
descriptor = KeypointDescriptor()

# Define the complete model and load the weights from the pretrained SuperPoint model
fsuperpoint = FlexibleSuperPoint(encoder, detector=detector, descriptor=descriptor).eval()
fsuperpoint.load("https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth", agglomerated=True)

# Read an image
img = imread("path/to/image")

# Use OpenCV interface to get the keypoints and their descriptors
keypoints, descriptors = fsuperpoint.detectAndCompute(img)
np_keypoints = np.array([(kp.pt[0], kp.pt[1]) for kp in keypoints])

plt.figure()
plt.imshow(img)
plt.plot(np_keypoints[:, 0], np_keypoints[:, 1], "r.")
plt.show()
```