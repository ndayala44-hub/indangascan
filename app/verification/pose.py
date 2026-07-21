"""
Head pose estimation from face landmarks (industry-standard approach).

Six stable MediaPipe FaceMesh landmarks are matched against a generic 3D
face model and solved with cv2.solvePnP; the rotation vector is converted
via Rodrigues and decomposed into Euler angles. This yields metric head
pose in degrees - robust to distance from camera, face size, and lateral
translation, unlike 2D landmark-ratio proxies.

Sign conventions (pinned by a synthetic projection round-trip in the test
suite, so they cannot silently drift):

    yaw   > 0  : subject turns their head to THEIR LEFT
                 (nose moves toward image-right in un-mirrored frames)
    pitch > 0  : subject looks UP
    roll       : in-plane tilt (unused by challenges)
"""

import math

import cv2
import numpy as np

# MediaPipe FaceMesh landmark indices for the six model points.
POSE_LANDMARKS = {
    "nose_tip": 1,
    "chin": 152,
    "right_eye_outer": 33,    # subject's right (image-left when un-mirrored)
    "left_eye_outer": 263,    # subject's left
    "mouth_right": 61,
    "mouth_left": 291,
}

# Generic 3D face model (millimeters). Axes: +X = subject's left,
# +Y = up, +Z = toward the camera.
MODEL_POINTS = np.array([
    [0.0, 0.0, 0.0],          # nose tip
    [0.0, -63.6, -12.5],      # chin
    [-43.3, 32.7, -26.0],     # right eye outer corner
    [43.3, 32.7, -26.0],      # left eye outer corner
    [-28.9, -28.9, -24.1],    # mouth right corner
    [28.9, -28.9, -24.1],     # mouth left corner
], dtype=np.float64)


def camera_matrix(width: int, height: int) -> np.ndarray:
    """Pinhole approximation: focal length = image width, center = middle."""
    focal = float(width)
    return np.array([
        [focal, 0, width / 2.0],
        [0, focal, height / 2.0],
        [0, 0, 1],
    ], dtype=np.float64)


def estimate_pose(image_points: np.ndarray, width: int, height: int):
    """
    image_points: (6, 2) pixel coordinates in POSE_LANDMARKS order.
    Returns (yaw_deg, pitch_deg, roll_deg) or None if the solve fails.
    """
    pts = np.asarray(image_points, dtype=np.float64).reshape(6, 2)
    ok, rvec, _ = cv2.solvePnP(
        MODEL_POINTS, pts, camera_matrix(width, height),
        np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    rot, _ = cv2.Rodrigues(rvec)

    # Decompose R (model->camera). Camera axes: X right, Y down, Z forward.
    # sy guards the gimbal-lock singularity.
    sy = math.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2)
    if sy > 1e-6:
        pitch_r = math.atan2(rot[2, 1], rot[2, 2])
        yaw_r = math.atan2(-rot[2, 0], sy)
        roll_r = math.atan2(rot[1, 0], rot[0, 0])
    else:
        pitch_r = math.atan2(-rot[1, 2], rot[1, 1])
        yaw_r = math.atan2(-rot[2, 0], sy)
        roll_r = 0.0

    # Negation maps the decomposition into the documented subject-centric
    # convention (validated by the synthetic round-trip test).
    yaw = -math.degrees(yaw_r)
    pitch = math.degrees(pitch_r)
    roll = math.degrees(roll_r)

    # Map into the documented subject-centric convention. The generic model
    # faces the camera, so raw pitch hovers near +/-180; normalize it.
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180
    return yaw, pitch, roll
