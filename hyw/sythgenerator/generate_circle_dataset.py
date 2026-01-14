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


def _sample_episode_macro_actions(rng: np.random.Generator) -> tuple[str, str]:
    """
    Pick a "big direction" for the episode:
    - macro_move: one of WASD (no empty)
    - macro_view: one of LR/LL/LU/LD (no empty)
    """
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
    macro_move, macro_view = _sample_episode_macro_actions(rng)
    move_alts = {"W": ["A", "D"], "S": ["A", "D"], "A": ["W", "S"], "D": ["W", "S"]}
    view_alts = {"LR": ["LU", "LD"], "LL": ["LU", "LD"], "LU": ["LR", "LL"], "LD": ["LR", "LL"]}
    # Probabilities tuned to: keep moving/turning, but with small jitter.
    move_empty_prob = 0.12
    move_macro_prob = 0.78
    move_alt_prob = 0.10
    view_empty_prob = 0.18
    view_macro_prob = 0.72
    view_alt_prob = 0.10
    max_idle_move = 3  # if we had no movement for this many consecutive frames, force macro_move
    idle_move_count = 0

    for t in range(num_frames):
        if t == 0:
            move_action = ""
            view_action = ""
        else:
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

            if move_action == "":
                idle_move_count += 1
                if idle_move_count >= max_idle_move:
                    move_action = macro_move
                    idle_move_count = 0
            else:
                idle_move_count = 0

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


