# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class WorldState:
    x: float = 0.0
    y: float = 0.0
    yaw_rad: float = 0.0
    pitch_rad: float = 0.0


def hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    """h in [0,1). s,v in [0,1]. returns 0-255 ints."""
    h = float(h % 1.0)
    s = float(np.clip(s, 0.0, 1.0))
    v = float(np.clip(v, 0.0, 1.0))
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return int(r * 255), int(g * 255), int(b * 255)


def apply_action(
    state: WorldState,
    move_action: str,
    view_action: str,
    move_step: float = 0.08,
    yaw_step_deg: float = 3.0,
    pitch_step_deg: float = 2.0,
    pitch_limit_deg: float = 35.0,
) -> None:
    # Movement on XY plane
    if "W" in move_action and "S" not in move_action:
        state.y += move_step
    if "S" in move_action and "W" not in move_action:
        state.y -= move_step
    if "D" in move_action and "A" not in move_action:
        state.x += move_step
    if "A" in move_action and "D" not in move_action:
        state.x -= move_step

    # View control
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


def render_frame(
    state: WorldState,
    width: int,
    height: int,
    circle_radius_px: int = 32,
    move_action: str = "",
    view_action: str = "",
) -> np.ndarray:
    """
    Render a simple 2D view:
    - Circle position controlled by WASD (state.x/state.y).
    - Circle color controlled by camera yaw/pitch (state.yaw_rad/state.pitch_rad).
    """
    img = Image.new("RGB", (width, height), (20, 20, 24))
    draw = ImageDraw.Draw(img)

    # Map world coords to pixels (centered)
    scale = min(width, height) * 0.18
    cx = width / 2.0 + state.x * scale
    cy = height / 2.0 - state.y * scale

    # Color depends on yaw (hue) and pitch (value)
    hue = (state.yaw_rad / (2.0 * math.pi)) % 1.0
    val = 0.65 + 0.35 * math.sin(state.pitch_rad)
    rgb = hsv_to_rgb(hue, 0.9, float(np.clip(val, 0.2, 1.0)))

    x0 = cx - circle_radius_px
    y0 = cy - circle_radius_px
    x1 = cx + circle_radius_px
    y1 = cy + circle_radius_px
    draw.ellipse([x0, y0, x1, y1], fill=rgb, outline=(250, 250, 250), width=2)

    # HUD text
    txt = (
        f"move={move_action or '-'} view={view_action or '-'} "
        f"yaw={math.degrees(state.yaw_rad):.1f} pitch={math.degrees(state.pitch_rad):.1f}"
    )
    draw.rectangle([8, 8, width - 8, 34], fill=(0, 0, 0))
    draw.text((12, 12), txt, fill=(230, 230, 230))

    return np.asarray(img, dtype=np.uint8)


