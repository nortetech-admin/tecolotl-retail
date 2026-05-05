"""
pose_detector.py — Tecolotl Retail
===================================
Parses raw IMX500 HigherHRNet metadata into structured Pose objects.

Based on the official Raspberry Pi picamera2 demo:
https://github.com/raspberrypi/picamera2/blob/main/examples/imx500/imx500_pose_estimation_higherhrnet_demo.py

The IMX500 pipeline internally uses postprocess_higherhrnet from:
picamera2.devices.imx500.postprocess_highernet

That function handles:
- Raw output tensor decoding
- Heatmap decoding
- Keypoint grouping per person
- Bounding box calculation
- Confidence threshold filtering

This module wraps that pipeline and exposes clean Pose objects
for use by person_tracker.py and the future shelf_attention module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from picamera2.devices.imx500 import IMX500
from picamera2.devices.imx500.postprocess_highernet import postprocess_higherhrnet

# ---------------------------------------------------------------------------
# COCO keypoint index constants
# Reference: https://github.com/raspberrypi/picamera2 — COCO keypoint format
# ---------------------------------------------------------------------------
KP_NOSE          = 0
KP_LEFT_EYE      = 1
KP_RIGHT_EYE     = 2
KP_LEFT_EAR      = 3
KP_RIGHT_EAR     = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER= 6
KP_LEFT_ELBOW    = 7
KP_RIGHT_ELBOW   = 8
KP_LEFT_WRIST    = 9
KP_RIGHT_WRIST   = 10
KP_LEFT_HIP      = 11
KP_RIGHT_HIP     = 12
KP_LEFT_KNEE     = 13
KP_RIGHT_KNEE    = 14
KP_LEFT_ANKLE    = 15
KP_RIGHT_ANKLE   = 16

# All 17 COCO keypoints — exposed for current and future use.
# Immediately useful: nose, shoulders, hips (orientation), eyes/ears (head direction)
# Future use: wrists + elbows for arm vector → detect if someone reaches toward a shelf
RETAIL_KEYPOINTS = {
    "nose":            KP_NOSE,
    "left_eye":        KP_LEFT_EYE,
    "right_eye":       KP_RIGHT_EYE,
    "left_ear":        KP_LEFT_EAR,
    "right_ear":       KP_RIGHT_EAR,
    "left_shoulder":   KP_LEFT_SHOULDER,
    "right_shoulder":  KP_RIGHT_SHOULDER,
    "left_elbow":      KP_LEFT_ELBOW,
    "right_elbow":     KP_RIGHT_ELBOW,
    "left_wrist":      KP_LEFT_WRIST,
    "right_wrist":     KP_RIGHT_WRIST,
    "left_hip":        KP_LEFT_HIP,
    "right_hip":       KP_RIGHT_HIP,
    "left_knee":       KP_LEFT_KNEE,
    "right_knee":      KP_RIGHT_KNEE,
    "left_ankle":      KP_LEFT_ANKLE,
    "right_ankle":     KP_RIGHT_ANKLE,
}

# Default window size — must match the IMX500 inference resolution
WINDOW_SIZE_H_W = (480, 640)

# Default confidence threshold (matches the official demo default)
DEFAULT_DETECTION_THRESHOLD = 0.3


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Keypoint:
    x: float
    y: float
    confidence: float

    def is_valid(self, min_confidence: float = 0.3) -> bool:
        """Returns True if this keypoint has enough confidence to be used."""
        return self.confidence >= min_confidence


@dataclass
class Pose:
    """
    A single detected person with 17 COCO keypoints, a bounding box, and a score.

    keypoints: list of 17 Keypoint objects in COCO order.
    box:       [x1, y1, x2, y2] in image coordinates.
    score:     overall detection confidence from HigherHRNet.
    """
    keypoints: list[Keypoint]
    box: np.ndarray
    score: float

    def get(self, index: int) -> Keypoint:
        """Return keypoint by COCO index."""
        return self.keypoints[index]

    def retail_keypoints(self) -> dict[str, Keypoint]:
        """Return only the keypoints relevant for shelf attention."""
        return {name: self.keypoints[idx] for name, idx in RETAIL_KEYPOINTS.items()}

    @property
    def nose(self) -> Keypoint:
        return self.keypoints[KP_NOSE]

    @property
    def left_shoulder(self) -> Keypoint:
        return self.keypoints[KP_LEFT_SHOULDER]

    @property
    def right_shoulder(self) -> Keypoint:
        return self.keypoints[KP_RIGHT_SHOULDER]

    @property
    def left_elbow(self) -> Keypoint:
        return self.keypoints[KP_LEFT_ELBOW]

    @property
    def right_elbow(self) -> Keypoint:
        return self.keypoints[KP_RIGHT_ELBOW]

    @property
    def left_wrist(self) -> Keypoint:
        return self.keypoints[KP_LEFT_WRIST]

    @property
    def right_wrist(self) -> Keypoint:
        return self.keypoints[KP_RIGHT_WRIST]

    @property
    def left_hip(self) -> Keypoint:
        return self.keypoints[KP_LEFT_HIP]

    @property
    def right_hip(self) -> Keypoint:
        return self.keypoints[KP_RIGHT_HIP]


# ---------------------------------------------------------------------------
# Core parsing function
# ---------------------------------------------------------------------------

def get_poses(
    metadata: dict,
    imx500: IMX500,
    window_size: tuple[int, int] = WINDOW_SIZE_H_W,
    detection_threshold: float = DEFAULT_DETECTION_THRESHOLD,
) -> list[Pose]:
    """
    Parse IMX500 metadata into a list of Pose objects.

    This is the primary interface for downstream consumers
    (person_tracker.py, shelf_attention.py).

    Args:
        metadata:            Raw metadata dict from request.get_metadata()
        imx500:              IMX500 device instance (used to extract output tensors)
        window_size:         (height, width) of the inference window
        detection_threshold: Minimum confidence to include a detection

    Returns:
        List of Pose objects. Empty list if no detections or no tensor output.

    Usage:
        poses = get_poses(request.get_metadata(), imx500)
        for pose in poses:
            print(pose.score, pose.nose.x, pose.nose.y)
    """
    np_outputs = imx500.get_outputs(metadata=metadata, add_batch=True)

    if np_outputs is None:
        return []

    keypoints_raw, scores, boxes = postprocess_higherhrnet(
        outputs=np_outputs,
        img_size=window_size,
        img_w_pad=(0, 0),
        img_h_pad=(0, 0),
        detection_threshold=detection_threshold,
        network_postprocess=True,
    )

    if scores is None or len(scores) == 0:
        return []

    # keypoints_raw shape: (N, 17, 3) — [x, y, confidence] per keypoint per person
    keypoints_array = np.reshape(
        np.stack(keypoints_raw, axis=0), (len(scores), 17, 3)
    )

    poses = []
    for i, score in enumerate(scores):
        kps = [
            Keypoint(x=float(kp[0]), y=float(kp[1]), confidence=float(kp[2]))
            for kp in keypoints_array[i]
        ]
        poses.append(Pose(
            keypoints=kps,
            box=np.array(boxes[i]),
            score=float(score),
        ))

    return poses


# ---------------------------------------------------------------------------
# Debug utility — print keypoints for a single pose
# ---------------------------------------------------------------------------

def print_pose(pose: Pose) -> None:
    """Print a human-readable summary of a Pose. Useful during development."""
    print(f"  Score: {pose.score:.2f}  Box: {pose.box}")
    for name, kp in pose.retail_keypoints().items():
        status = "✓" if kp.is_valid() else "✗"
        print(f"  [{status}] {name:20s}  x={kp.x:.1f}  y={kp.y:.1f}  conf={kp.confidence:.2f}")