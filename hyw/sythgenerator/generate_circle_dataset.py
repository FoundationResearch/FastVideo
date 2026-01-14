# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio
import numpy as np

from hyw.sythgenerator.pose_math import CameraPose, make_intrinsic, orbit_camera_w2c
from hyw.sythgenerator.render_circle_world import WorldState, apply_action, render_frame


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _choice(rng: np.random.Generator, items: list[str], probs: list[float]) -> str:
    idx = int(rng.choice(len(items), p=np.array(probs, dtype=np.float64)))
    return items[idx]


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

    move_actions = ["", "W", "A", "S", "D"]
    move_probs = [0.45, 0.14, 0.14, 0.14, 0.13]
    view_actions = ["", "LR", "LL", "LU", "LD"]
    view_probs = [0.55, 0.12, 0.12, 0.11, 0.10]

    for t in range(num_frames):
        if t == 0:
            move_action = ""
            view_action = ""
        else:
            move_action = _choice(rng, move_actions, move_probs)
            view_action = _choice(rng, view_actions, view_probs)

        apply_action(
            state,
            move_action=move_action,
            view_action=view_action,
            move_step=move_step,
            yaw_step_deg=yaw_step_deg,
            pitch_step_deg=pitch_step_deg,
        )

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
        },
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate synthetic circle+action+pose videos for HY-WorldPlay.")
    p.add_argument("--out_root", type=str, default="/home/hao_lab/alex/FastVideo/hyw/data/sythcircle_v0")
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--num_frames", type=int, default=64)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--camera_radius", type=float, default=3.0)
    p.add_argument("--fov_deg", type=float, default=70.0)
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


