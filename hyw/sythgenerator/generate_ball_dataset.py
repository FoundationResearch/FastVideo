# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio
import numpy as np

from hyw.sythgenerator.pose_math import make_intrinsic
from hyw.sythgenerator.render_ball_world import (
    BallWorldState,
    apply_action_3d,
    precompute_camera_rays_cam,
    render_ball_frame,
    yaw_pitch_look_at,
    w2c_from_camera_pose,
)


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
    return Path.cwd().resolve()


def _default_out_root() -> str:
    repo_root = _find_repo_root(Path(__file__).resolve())
    return str((repo_root / "hyw" / "data" / "sythball_v0").resolve())


def _choice(rng: np.random.Generator, items: list[str], probs: list[float]) -> str:
    idx = int(rng.choice(len(items), p=np.array(probs, dtype=np.float64)))
    return items[idx]


def _sample_episode_macro_actions(rng: np.random.Generator) -> tuple[str, str]:
    macro_move = str(rng.choice(["W", "A", "S", "D"]))
    macro_view = str(rng.choice(["LR", "LL", "LU", "LD"]))
    return macro_move, macro_view


def _micro_action(
    rng: np.random.Generator,
    macro: str,
    *,
    empty_prob: float,
    macro_prob: float,
    alt_prob: float,
    alts: dict[str, list[str]],
) -> str:
    r = float(rng.random())
    if r < empty_prob:
        return ""
    r -= empty_prob
    if r < macro_prob:
        return macro
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
    fov_deg: float = 70.0,
    move_step: float = 0.08,
    yaw_step_deg: float = 3.0,
    pitch_step_deg: float = 2.0,
    macro_period: int = 12,
    world_bounds_xz: tuple[float, float, float, float] = (-2.0, 2.0, -2.0, 3.0),
) -> dict:
    """
    Generates a simple 3D scene ("sythball"):
      - ground plane + a ball (sphere) on the plane
      - camera translation (WASD) + camera yaw/pitch (view_action) are truly rendered

    Outputs compatible with HY-WorldPlay:
      - video.mp4
      - pose.json (per-frame intrinsic+w2c)
      - action.json (per-frame move_action/view_action)
    """
    rng = np.random.default_rng(seed)

    K = make_intrinsic(width=width, height=height, fov_deg=fov_deg)
    rays_cam = precompute_camera_rays_cam(width, height, K)

    state = BallWorldState()
    sphere_center = np.array([0.0, 0.35, 0.8], dtype=np.float32)
    # Make the first frame look at the ball.
    cam_pos0 = np.array([state.cam_x, state.cam_y, state.cam_z], dtype=np.float32)
    state.yaw_rad, state.pitch_rad = yaw_pitch_look_at(cam_pos0, sphere_center)
    frames: list[np.ndarray] = []
    pose_json: dict[str, dict] = {}
    action_json: dict[str, dict] = {}

    macro_move, macro_view = _sample_episode_macro_actions(rng)
    move_alts = {"W": ["WD", "WA"], "S": ["SD", "SA"], "D": ["WD", "SD"], "A": ["WA", "SA"]}
    view_alts = {"LR": ["LU", "LD"], "LL": ["LU", "LD"], "LU": ["LR", "LL"], "LD": ["LR", "LL"]}

    move_empty_prob = 0.15
    move_macro_prob = 0.75
    move_alt_prob = 0.10
    view_empty_prob = 0.18
    view_macro_prob = 0.72
    view_alt_prob = 0.10

    for t in range(num_frames):
        if t == 0:
            move_action = ""
            view_action = ""
        else:
            # Occasionally change the macro actions to avoid trivial straight lines
            if macro_period > 0 and (t % macro_period) == 0:
                macro_move, macro_view = _sample_episode_macro_actions(rng)

            move_action = _micro_action(
                rng,
                macro_move,
                empty_prob=move_empty_prob,
                macro_prob=move_macro_prob,
                alt_prob=move_alt_prob,
                alts=move_alts,
            )
            view_action = _micro_action(
                rng,
                macro_view,
                empty_prob=view_empty_prob,
                macro_prob=view_macro_prob,
                alt_prob=view_alt_prob,
                alts=view_alts,
            )

        apply_action_3d(
            state,
            move_action=move_action,
            view_action=view_action,
            move_step=move_step,
            yaw_step_deg=yaw_step_deg,
            pitch_step_deg=pitch_step_deg,
            world_bounds_xz=world_bounds_xz,
        )

        cam_pos = np.array([state.cam_x, state.cam_y, state.cam_z], dtype=np.float32)
        w2c = w2c_from_camera_pose(cam_pos, state.yaw_rad, state.pitch_rad)

        pose_json[str(t)] = {
            "intrinsic": K.tolist(),
            "w2c": w2c.tolist(),
            "K": K.tolist(),
            "extrinsic": w2c.tolist(),
        }
        action_json[str(t)] = {"move_action": move_action, "view_action": view_action}

        frame = render_ball_frame(
            state,
            width=width,
            height=height,
            K=K,
            rays_cam=rays_cam,
            sphere_center=(float(sphere_center[0]), float(sphere_center[1]), float(sphere_center[2])),
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
            "fov_deg": fov_deg,
            "move_step": move_step,
            "yaw_step_deg": yaw_step_deg,
            "pitch_step_deg": pitch_step_deg,
            "macro_period": macro_period,
            "world_bounds_xz": list(world_bounds_xz),
            "scene": "sythball_plane+sphere",
        },
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate synthetic 3D sythball videos for HY-WorldPlay.")
    p.add_argument("--out_root", type=str, default=_default_out_root())
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--num_frames", type=int, default=64)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fov_deg", type=float, default=70.0)
    p.add_argument("--move_step", type=float, default=0.08)
    p.add_argument("--yaw_step_deg", type=float, default=3.0)
    p.add_argument("--pitch_step_deg", type=float, default=2.0)
    p.add_argument(
        "--macro_period",
        type=int,
        default=12,
        help="Change macro move/view direction every N frames (t>0). Set 0 to disable.",
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
            fov_deg=args.fov_deg,
            move_step=args.move_step,
            yaw_step_deg=args.yaw_step_deg,
            pitch_step_deg=args.pitch_step_deg,
            macro_period=args.macro_period,
        )
        entry["id"] = f"{args.split}_{i:05d}"
        entry["split"] = args.split
        entry["text"] = "a 3D scene with a ball on a ground plane; camera moves with WASD and rotates by view actions"
        manifest.append(entry)

    manifest_path = out_root / f"manifest_raw_{args.split}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[OK] wrote {len(manifest)} samples to {split_dir}")
    print(f"[OK] manifest: {manifest_path}")


if __name__ == "__main__":
    main()


