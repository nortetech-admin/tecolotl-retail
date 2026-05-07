"""
shelf_attention.py — Tecolotl Retail
=====================================
Determines whether a detected person is facing the shelf
and which zone of the shelf they are positioned in.

Zones are defined by dividing the image width into N vertical sections
using evenly-spaced boundary lines. The person's horizontal center
(midpoint of their shoulders) determines their zone.

Usage:
    from shelf_attention import is_facing_shelf, get_shelf_zone, ZONE_COUNT
"""

from __future__ import annotations

from pose_detector import Pose

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum shoulder width in pixels to consider the pose reliable for orientation.
# If the detected shoulder width is below this, the person is likely turned sideways.
MIN_SHOULDER_WIDTH: int = 40

# Number of vertical zones the shelf is divided into.
# Zone 1 = leftmost, Zone N = rightmost.
ZONE_COUNT: int = 4

# Image width in pixels — must match the IMX500 inference resolution.
# Used to calculate zone boundaries.
IMAGE_WIDTH: int = 640


# ---------------------------------------------------------------------------
# Zone helpers
# ---------------------------------------------------------------------------

def get_zone_boundaries(
    zone_count: int = ZONE_COUNT,
    image_width: int = IMAGE_WIDTH,
) -> list[float]:
    """
    Return a list of (zone_count - 1) x-coordinate boundaries that divide
    the image into equal vertical zones.

    Example for 4 zones on a 640px image:
        [160.0, 320.0, 480.0]
    """
    return [image_width * i / zone_count for i in range(1, zone_count)]


def get_shelf_zone(
    x: float,
    zone_count: int = ZONE_COUNT,
    image_width: int = IMAGE_WIDTH,
) -> int:
    """
    Return the 1-based zone index for a given x coordinate.

    Args:
        x:           Horizontal position in pixels.
        zone_count:  Total number of zones.
        image_width: Width of the image in pixels.

    Returns:
        Integer from 1 to zone_count (inclusive).
    """
    zone_width = image_width / zone_count
    zone = int(x / zone_width) + 1
    # Clamp to valid range in case x == image_width exactly
    return min(zone, zone_count)


# ---------------------------------------------------------------------------
# Orientation detection
# ---------------------------------------------------------------------------

def is_facing_shelf(
    pose: Pose,
    min_shoulder_width: int = MIN_SHOULDER_WIDTH,
) -> bool | None:
    """
    Determine whether a person is facing the shelf (back to camera).

    In a retail aisle where the shelf is in front of the person:
    - If the person faces the shelf, their back is to the camera.
      The IMX500/HigherHRNet model labels shoulders from the camera's
      perspective, so left_shoulder.x < right_shoulder.x means the
      person's body is oriented away from the camera -> facing the shelf.
    - If left_shoulder.x > right_shoulder.x, the person faces the camera
      -> not looking at the shelf.

    Args:
        pose:               Detected Pose object.
        min_shoulder_width: Minimum pixel distance between shoulders.
                            Below this threshold the orientation is
                            considered inconclusive (person sideways).

    Returns:
        True  -- person is facing the shelf (back to camera).
        False -- person is facing the camera (front to camera).
        None  -- inconclusive (keypoints missing or shoulder width too small).
    """
    ls = pose.left_shoulder
    rs = pose.right_shoulder

    if not ls.is_valid() or not rs.is_valid():
        return None

    shoulder_width = abs(rs.x - ls.x)
    if shoulder_width < min_shoulder_width:
        return None

    return ls.x < rs.x


# ---------------------------------------------------------------------------
# Combined analysis
# ---------------------------------------------------------------------------

def analyze_pose(
    pose: Pose,
    min_shoulder_width: int = MIN_SHOULDER_WIDTH,
    zone_count: int = ZONE_COUNT,
    image_width: int = IMAGE_WIDTH,
) -> dict:
    """
    Return a full analysis dict for a single Pose:

        {
            "facing_shelf": True | False | None,
            "zone":         1..N | None,
            "shoulder_center_x": float | None,
        }

    Zone is None when there are no valid shoulder keypoints.
    """
    facing = is_facing_shelf(pose, min_shoulder_width)

    ls = pose.left_shoulder
    rs = pose.right_shoulder

    if ls.is_valid() and rs.is_valid():
        center_x = (ls.x + rs.x) / 2.0
        zone = get_shelf_zone(center_x, zone_count, image_width)
    else:
        center_x = None
        zone = None

    return {
        "facing_shelf": facing,
        "zone": zone,
        "shoulder_center_x": center_x,
    }