# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio
import numpy as np

from hyw.sythgenerator.pose_math import CameraPose, make_intrinsic, orbit_camera_w2c
from hyw.sythgenerator.render_circle_world import WorldState, apply_action, render_frame


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _find_repo_root(start: Path) -> Path:
    """
    Find FastVideo repo root by searching upwards for pyproject.toml.
    Keeps defaults portable across machines (no hard-coded /home/hao_lab/...).
    """
    cur = start.resolve()
    for _ in range(10):
        if (cur / "pyproject.toml").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    # Fallback: current working directory
    return Path.cwd().resolve()


def _default_out_root() -> str:
    repo_root = _find_repo_root(Path(__file__).resolve())
    return str((repo_root / "hyw" / "data" / "sythcircle_v0").resolve())


def _choice(rng: np.random.Generator, items: list[str], probs: list[float]) -> str:
    idx = int(rng.choice(len(items), p=np.array(probs, dtype=np.float64)))
    return items[idx]


def _sample_episode_macro_actions(rng: np.random.Generator) -> tuple[str, str]:
    """
    Pick a "big direction" for the episode:
    - macro_move: one of WASD (no empty)
    - macro_view: one of LR/LL/LU/LD (no empty)
    """
    macro_move = str(rng.choice(["W", "A", "S", "D"]))
    macro_view = str(rng.choice(["LR", "LL", "LU", "LD"]))
    return macro_move, macro_view


def _sample_macro_move_angle_rad(rng: np.random.Generator) -> float:
    """Sample a continuous move direction angle (radians) in the XY plane."""
    return float(rng.uniform(0.0, 2.0 * math.pi))


def _angle_to_move_action(angle_rad: float, *, deadzone: float = 0.25) -> str:
    """
    Convert a continuous direction (angle) into a WASD-like string for logging/compat.
    This keeps the output format stable while motion itself can still be continuous.
    """
    vx = math.cos(angle_rad)
    vy = math.sin(angle_rad)
    s = ""
    if vy > deadzone:
        s += "W"
    elif vy < -deadzone:
        s += "S"
    if vx > deadzone:
        s += "D"
    elif vx < -deadzone:
        s += "A"
    return s


def _micro_action(
    rng: np.random.Generator,
    macro: str,
    *,
    empty_prob: float,
    macro_prob: float,
    alt_prob: float,
    alts: dict[str, list[str]],
) -> str:
    """
    Sample an action with a macro direction + small perturbations.
    """
    r = float(rng.random())
    if r < empty_prob:
        return ""
    r -= empty_prob
    if r < macro_prob:
        return macro
    # small perturbation
    if r < macro_prob + alt_prob:
        return str(rng.choice(alts.get(macro, [macro])))
    return macro


def generate_one_episode(
    out_dir: Path,
    *,
    num_frames: int,
    fps: int,
    width: int,
    height: int,
    seed: int,
    camera_radius: float = 3.0,
    fov_deg: float = 70.0,
    move_step: float = 0.08,
    yaw_step_deg: float = 3.0,
    pitch_step_deg: float = 2.0,
    macro_period: int = 8,
    move_dir_jitter_deg: float = 8.0,
    hold_action_frames: int = 4,
    circle_radius_px: int = 32,
) -> dict:
    """
    Generates:
      - video.mp4 (uint8 RGB)
      - pose.json  (keys: "0"...; fields include intrinsic+w2c, plus compat K+extrinsic)
      - action.json (keys: "0"...; fields include move_action/view_action)
    """
    rng = np.random.default_rng(seed)

    K = make_intrinsic(width=width, height=height, fov_deg=fov_deg)

    state = WorldState(x=0.0, y=0.0, yaw_rad=0.0, pitch_rad=0.0)
    frames: list[np.ndarray] = []
    pose_json: dict[str, dict] = {}
    action_json: dict[str, dict] = {}

    # Episode-level "big direction", then per-frame micro adjustments.
    # - movement: continuous direction (any angle) in XY plane, changed every `macro_period` frames
    # - view: keep 4-way discrete actions for simplicity
    _, macro_view = _sample_episode_macro_actions(rng)
    macro_move_angle = _sample_macro_move_angle_rad(rng)
    view_alts = {"LR": ["LU", "LD"], "LL": ["LU", "LD"], "LU": ["LR", "LL"], "LD": ["LR", "LL"]}

    # We always apply continuous movement for t>0. `move_action` is a compact label only.
    view_empty_prob = 0.18
    view_macro_prob = 0.72
    view_alt_prob = 0.10

    # Screen-edge boundary in world coords (consistent with render_frame mapping).
    scale = min(width, height) * 0.18
    max_x = (width / 2.0 - float(circle_radius_px)) / scale
    max_y = (height / 2.0 - float(circle_radius_px)) / scale
    world_bounds = (-max_x, max_x, -max_y, max_y)

    if hold_action_frames <= 0:
        raise ValueError("--hold_action_frames must be a positive integer")

    # Hold move/view actions constant inside each `hold_action_frames` block to better align with
    # latent-level conditioning (1 latent ~= 4 frames).
    cur_move_action = ""
    cur_view_action = ""
    cur_move_dir_xy = (0.0, 0.0)

    for t in range(num_frames):
        if t == 0:
            cur_move_action = ""
            cur_view_action = ""
            cur_move_dir_xy = (0.0, 0.0)
        else:
            if (t % hold_action_frames) == 0:
                # Change macro movement direction every N frames (t>0), aligned to block boundaries.
                if macro_period > 0 and (t % macro_period) == 0:
                    macro_move_angle = _sample_macro_move_angle_rad(rng)

                # Jitter around macro direction (sampled once per block).
                jitter = float(
                    rng.normal(loc=0.0, scale=math.radians(move_dir_jitter_deg))
                )
                move_angle = float((macro_move_angle + jitter) % (2.0 * math.pi))
                cur_move_dir_xy = (math.cos(move_angle), math.sin(move_angle))
                cur_move_action = _angle_to_move_action(move_angle)
                cur_view_action = _micro_action(
                    rng,
                    macro_view,
                    empty_prob=view_empty_prob,
                    macro_prob=view_macro_prob,
                    alt_prob=view_alt_prob,
                    alts=view_alts,
                )

        move_action = cur_move_action
        view_action = cur_view_action
        move_dir_xy = cur_move_dir_xy

        prev_x, prev_y = state.x, state.y
        apply_action(
            state,
            move_action=move_action,
            view_action=view_action,
            move_step=move_step,
            yaw_step_deg=yaw_step_deg,
            pitch_step_deg=pitch_step_deg,
            move_dir_xy=move_dir_xy,
            world_bounds=world_bounds,
        )
        # If we hit the screen edge and couldn't move, resample macro direction once to avoid getting stuck.
        if t > 0 and state.x == prev_x and state.y == prev_y:
            macro_move_angle = _sample_macro_move_angle_rad(rng)

        # Camera pose uses the SAME yaw/pitch that drives color -> matches requirement.
        w2c = orbit_camera_w2c(CameraPose(yaw_rad=state.yaw_rad, pitch_rad=state.pitch_rad, radius=camera_radius))

        # Store full-frame pose/action with string keys.
        pose_json[str(t)] = {
            "intrinsic": K.tolist(),
            "w2c": w2c.tolist(),
            # Extra compat fields (some code/assets use these names)
            "K": K.tolist(),
            "extrinsic": w2c.tolist(),
        }
        action_json[str(t)] = {
            "move_action": move_action,
            "view_action": view_action,
        }

        frame = render_frame(
            state,
            width=width,
            height=height,
            move_action=move_action,
            view_action=view_action,
        )
        frames.append(frame)

    _ensure_dir(out_dir)
    video_path = out_dir / "video.mp4"
    pose_path = out_dir / "pose.json"
    action_path = out_dir / "action.json"

    with imageio.get_writer(str(video_path), fps=fps, format="mp4") as writer:
        for fr in frames:
            writer.append_data(fr)

    pose_path.write_text(json.dumps(pose_json, indent=2), encoding="utf-8")
    action_path.write_text(json.dumps(action_json, indent=2), encoding="utf-8")

    return {
        "video_path": str(video_path),
        "pose_path": str(pose_path),
        "action_path": str(action_path),
        "meta": {
            "seed": seed,
            "num_frames": num_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "camera_radius": camera_radius,
            "fov_deg": fov_deg,
            "move_step": move_step,
            "yaw_step_deg": yaw_step_deg,
            "pitch_step_deg": pitch_step_deg,
            "macro_period": macro_period,
            "move_dir_jitter_deg": move_dir_jitter_deg,
            "circle_radius_px": circle_radius_px,
        },
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate synthetic circle+action+pose videos for HY-WorldPlay.")
    p.add_argument("--out_root", type=str, default=_default_out_root())
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--num_frames", type=int, default=64)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--camera_radius", type=float, default=3.0)
    p.add_argument("--fov_deg", type=float, default=70.0)
    p.add_argument(
        "--macro_period",
        type=int,
        default=8,
        help="Change movement macro direction every N frames (t>0).",
    )
    p.add_argument(
        "--move_dir_jitter_deg",
        type=float,
        default=8.0,
        help="Per-frame angular jitter (deg) around macro direction.",
    )
    p.add_argument(
        "--hold_action_frames",
        type=int,
        default=4,
        help="Hold move/view action constant for this many frames (default: 4). "
        "Use 4 to align with 1 latent ~= 4 frames.",
    )
    p.add_argument(
        "--circle_radius_px",
        type=int,
        default=32,
        help="Circle radius in pixels (used for screen-edge boundary).",
    )
    args = p.parse_args(argv)

    out_root = Path(args.out_root).expanduser().resolve()
    split_dir = out_root / args.split
    _ensure_dir(split_dir)

    manifest: list[dict] = []
    for i in range(args.num_samples):
        sample_dir = split_dir / f"sample_{i:05d}"
        entry = generate_one_episode(
            sample_dir,
            num_frames=args.num_frames,
            fps=args.fps,
            width=args.width,
            height=args.height,
            seed=args.seed + i,
            camera_radius=args.camera_radius,
            fov_deg=args.fov_deg,
            macro_period=args.macro_period,
            move_dir_jitter_deg=args.move_dir_jitter_deg,
            hold_action_frames=args.hold_action_frames,
            circle_radius_px=args.circle_radius_px,
        )
        entry["id"] = f"{args.split}_{i:05d}"
        entry["split"] = args.split
        entry["text"] = "a colored circle controlled by WASD; camera view angle changes its color"
        manifest.append(entry)

    manifest_path = out_root / f"manifest_raw_{args.split}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[OK] wrote {len(manifest)} samples to {split_dir}")
    print(f"[OK] manifest: {manifest_path}")


if __name__ == "__main__":
    main()


