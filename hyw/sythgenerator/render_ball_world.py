# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class BallWorldState:
    """
    Minimal 3D "worldplay" state:
    - camera translates on XZ plane (y fixed)
    - camera rotates by yaw/pitch
    """

    cam_x: float = 0.0
    cam_y: float = 1.2
    cam_z: float = -2.8
    yaw_rad: float = 0.0
    pitch_rad: float = math.radians(-12.0)


def _clamp(x: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    return np.clip(x, lo, hi)


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def camera_c2w_from_yaw_pitch(yaw: float, pitch: float) -> np.ndarray:
    """
    Camera-to-world rotation for a right-handed system:
    - camera axes in camera frame: x=right, y=up, z=forward
    - yaw around world Y
    - pitch around camera X (implemented as world X after yaw for simplicity)
    """
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)

    # Yaw around world Y.
    R_yaw = np.array(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        dtype=np.float32,
    )
    # Pitch around camera/local X. We approximate by rotating around world X after yaw.
    R_pitch = np.array(
        [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]],
        dtype=np.float32,
    )
    return (R_yaw @ R_pitch).astype(np.float32)


def w2c_from_camera_pose(cam_pos: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    """
    Build 4x4 world-to-camera from (cam_pos, yaw, pitch).
    """
    R_c2w = camera_c2w_from_yaw_pitch(yaw, pitch)  # (3,3)
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ cam_pos.reshape(3, 1)
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R_w2c
    w2c[:3, 3] = t_w2c[:, 0]
    return w2c


def apply_action_3d(
    state: BallWorldState,
    *,
    move_action: str,
    view_action: str,
    move_step: float = 0.08,
    yaw_step_deg: float = 3.0,
    pitch_step_deg: float = 2.0,
    pitch_limit_deg: float = 35.0,
    world_bounds_xz: tuple[float, float, float, float] | None = None,
) -> None:
    """
    Apply discrete WASD translation in WORLD axes (x/z) + view yaw/pitch.

    This matches HY-WorldPlay's "action_for_pe" heuristic that interprets translation direction
    in a fixed coordinate basis (x and z components).
    """
    dx = 0.0
    dz = 0.0
    if "W" in move_action and "S" not in move_action:
        dz += move_step
    if "S" in move_action and "W" not in move_action:
        dz -= move_step
    if "D" in move_action and "A" not in move_action:
        dx += move_step
    if "A" in move_action and "D" not in move_action:
        dx -= move_step

    if dx != 0.0 or dz != 0.0:
        new_x = state.cam_x + dx
        new_z = state.cam_z + dz
        if world_bounds_xz is not None:
            min_x, max_x, min_z, max_z = world_bounds_xz
            new_x = float(np.clip(new_x, min_x, max_x))
            new_z = float(np.clip(new_z, min_z, max_z))
        state.cam_x = new_x
        state.cam_z = new_z

    yaw_step = math.radians(yaw_step_deg)
    pitch_step = math.radians(pitch_step_deg)
    if view_action == "LR":
        state.yaw_rad += yaw_step
    elif view_action == "LL":
        state.yaw_rad -= yaw_step
    elif view_action == "LU":
        state.pitch_rad += pitch_step
    elif view_action == "LD":
        state.pitch_rad -= pitch_step

    pitch_limit = math.radians(pitch_limit_deg)
    state.pitch_rad = float(np.clip(state.pitch_rad, -pitch_limit, pitch_limit))


def _precompute_camera_rays_cam(width: int, height: int, K: np.ndarray) -> np.ndarray:
    """
    Returns ray directions in camera frame for each pixel: (H,W,3), not normalized.
    """
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    xs = (np.arange(width, dtype=np.float32) + 0.5 - cx) / fx
    ys = (np.arange(height, dtype=np.float32) + 0.5 - cy) / fy
    xx, yy = np.meshgrid(xs, ys)
    dirs = np.stack([xx, -yy, np.ones_like(xx)], axis=-1)  # y-down image to y-up camera
    return dirs.astype(np.float32)


def precompute_camera_rays_cam(width: int, height: int, K: np.ndarray) -> np.ndarray:
    """
    Public wrapper for precomputing per-pixel ray directions in camera frame.
    Returned array can be reused across frames as long as (W,H,K) are unchanged.
    """
    return _precompute_camera_rays_cam(width, height, K)


def render_ball_frame(
    state: BallWorldState,
    *,
    width: int,
    height: int,
    K: np.ndarray,
    rays_cam: np.ndarray,
    sphere_center: tuple[float, float, float] = (0.0, 0.35, 0.8),
    sphere_radius: float = 0.35,
    sun_dir: tuple[float, float, float] = (0.8, 1.6, 0.3),
) -> np.ndarray:
    """
    Vectorized tiny ray tracer:
    - infinite ground plane (y=0) with checkerboard
    - one sphere resting on the plane
    - lambert + small specular + sky background
    """
    # Camera pose
    cam_pos = np.array([state.cam_x, state.cam_y, state.cam_z], dtype=np.float32)
    R_c2w = camera_c2w_from_yaw_pitch(state.yaw_rad, state.pitch_rad)  # (3,3)

    # Rays in world
    dirs_world = rays_cam @ R_c2w.T  # (H,W,3)
    dirs_world = _normalize(dirs_world)
    o = cam_pos.reshape(1, 1, 3)
    d = dirs_world

    # Scene parameters
    c = np.array(sphere_center, dtype=np.float32).reshape(1, 1, 3)
    r = float(sphere_radius)
    L = _normalize(np.array(sun_dir, dtype=np.float32).reshape(1, 1, 3))

    # Sphere intersection: |o + t d - c|^2 = r^2
    oc = o - c
    b = 2.0 * np.sum(d * oc, axis=-1)  # (H,W)
    cc = np.sum(oc * oc, axis=-1) - r * r
    disc = b * b - 4.0 * cc
    hit_sphere = disc > 0.0
    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
    t_sphere = (-b - sqrt_disc) * 0.5
    hit_sphere = hit_sphere & (t_sphere > 1e-3)
    t_sphere = np.where(hit_sphere, t_sphere, np.inf).astype(np.float32)

    # Plane intersection: y=0
    dy = d[..., 1]
    t_plane = (0.0 - o[..., 1]) / np.where(np.abs(dy) < 1e-6, 1e-6, dy)
    hit_plane = (t_plane > 1e-3) & (dy < -1e-5)
    t_plane = np.where(hit_plane, t_plane, np.inf).astype(np.float32)

    # Choose nearest hit
    t = np.minimum(t_sphere, t_plane)
    hit_any = np.isfinite(t)

    p = o + d * t[..., None]  # (H,W,3)

    # Background (sky gradient)
    sky = np.zeros((height, width, 3), dtype=np.float32)
    tsky = _clamp(0.5 + 0.5 * d[..., 1], 0.0, 1.0)
    sky[..., 0] = 0.55 * (1.0 - tsky) + 0.10 * tsky
    sky[..., 1] = 0.70 * (1.0 - tsky) + 0.25 * tsky
    sky[..., 2] = 0.95 * (1.0 - tsky) + 0.55 * tsky

    # Shading init
    rgb = sky.copy()

    # Plane shading
    plane_mask = hit_plane & (t_plane <= t_sphere)
    if np.any(plane_mask):
        pp = p[plane_mask]
        # checkerboard on XZ
        freq = 2.0
        checks = (
            (np.floor(pp[:, 0] * freq) + np.floor(pp[:, 2] * freq)).astype(np.int32) & 1
        ).astype(np.float32)
        base0 = np.array([0.20, 0.20, 0.22], dtype=np.float32)
        base1 = np.array([0.55, 0.55, 0.60], dtype=np.float32)
        base = base0 * (1.0 - checks[:, None]) + base1 * checks[:, None]
        n = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        diff = np.maximum(0.0, np.sum(n.reshape(1, 3) * L.reshape(1, 3), axis=-1))
        col = base * (0.25 + 0.75 * diff)
        # distance fog
        dist = np.linalg.norm(pp - cam_pos.reshape(1, 3), axis=-1)
        fog = _clamp((dist - 3.0) / 6.0, 0.0, 1.0)
        col = col * (1.0 - fog[:, None]) + sky[plane_mask] * fog[:, None]
        rgb[plane_mask] = col

    # Sphere shading
    sphere_mask = hit_sphere & (t_sphere < t_plane)
    if np.any(sphere_mask):
        ps = p[sphere_mask]
        ns = _normalize(ps - np.array(sphere_center, dtype=np.float32).reshape(1, 3))
        base = np.array([0.85, 0.20, 0.18], dtype=np.float32)  # red ball
        ndotl = np.maximum(0.0, np.sum(ns * L.reshape(1, 3), axis=-1))
        # specular
        v = _normalize(cam_pos.reshape(1, 3) - ps)
        h = _normalize(L.reshape(1, 3) + v)
        spec = np.power(np.maximum(0.0, np.sum(ns * h, axis=-1)), 64.0)
        col = base * (0.18 + 0.82 * ndotl[:, None]) + 0.35 * spec[:, None]
        # simple ambient occlusion near ground contact
        ao = _clamp((ps[:, 1] - 0.05) / 0.25, 0.35, 1.0)
        col *= ao[:, None]
        rgb[sphere_mask] = col

    out = (_clamp(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return out


