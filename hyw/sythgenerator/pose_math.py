# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraPose:
    """Camera pose parameters in a simple orbit/look-at parameterization."""

    yaw_rad: float
    pitch_rad: float
    radius: float = 3.0


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0
    return v / n


def look_at_w2c(
    eye: np.ndarray,
    target: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float32),
    up: np.ndarray = np.array([0.0, 1.0, 0.0], dtype=np.float32),
) -> np.ndarray:
    """
    Build a 4x4 world-to-camera matrix (W2C) using a right-handed look-at convention.

    Returns:
        w2c: (4, 4) float32
    """
    eye = eye.astype(np.float32)
    target = target.astype(np.float32)
    up = up.astype(np.float32)

    forward = _normalize(target - eye)  # camera looks towards target
    right = _normalize(np.cross(forward, up))
    true_up = _normalize(np.cross(right, forward))

    # Camera-to-world rotation: columns are camera basis in world coordinates.
    # We define camera axes as: x=right, y=up, z=forward.
    R_c2w = np.stack([right, true_up, forward], axis=1)  # (3,3)
    t_c2w = eye.reshape(3, 1)

    # Invert to get w2c: R_w2c = R_c2w^T, t_w2c = -R^T * t
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w

    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R_w2c
    w2c[:3, 3] = t_w2c[:, 0]
    return w2c


def orbit_camera_w2c(pose: CameraPose) -> np.ndarray:
    """
    Create a camera orbiting around origin with given yaw/pitch.
    - yaw: rotation around world Y axis
    - pitch: rotation around world X axis (tilt up/down)
    """
    yaw = pose.yaw_rad
    pitch = pose.pitch_rad
    r = pose.radius

    # Spherical-ish parameterization around origin
    x = r * math.sin(yaw) * math.cos(pitch)
    y = r * math.sin(pitch)
    z = r * math.cos(yaw) * math.cos(pitch)
    eye = np.array([x, y, z], dtype=np.float32)
    return look_at_w2c(eye=eye)


def make_intrinsic(
    width: int,
    height: int,
    fov_deg: float = 70.0,
) -> np.ndarray:
    """
    Create a simple pinhole intrinsic matrix (3,3).
    HY-WorldPlay later normalizes fx/fy by cx/cy, so any plausible pinhole is OK.
    """
    fov = math.radians(fov_deg)
    fx = (width / 2.0) / math.tan(fov / 2.0)
    fy = (height / 2.0) / math.tan(fov / 2.0)
    cx = width / 2.0
    cy = height / 2.0
    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return K


