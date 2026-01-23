# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import imageio
import numpy as np

from hyw.sythgenerator.pose_math import make_intrinsic
from hyw.sythgenerator.render_ball_world import (
    BallWorldState,
    apply_action_3d,
    precompute_camera_rays_cam,
    render_ball_frame,
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
    hold_action_frames: int = 4,
    fixed_move_action: str | None = None,
    fixed_view_action: str | None = None,
    fixed_move_action_fn: Callable[[int], str] | None = None,
    # (min_x, max_x, min_z, max_z) in world coordinates.
    # Default enlarged 2x vs legacy (-2,2,-2,3) to reduce boundary clamping for short debug runs.
    world_bounds_xz: tuple[float, float, float, float] = (-4.0, 4.0, -4.0, 6.0),
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

    # For fixed-direction debug datasets (e.g. 4dir), enlarge bounds automatically so motion
    # won't get clamped (notably: default cam_z=-2.0 equals default min_z=-2.0, so 'S' can look static).
    if fixed_move_action is not None or fixed_move_action_fn is not None:
        min_x, max_x, min_z, max_z = world_bounds_xz
        # Rough upper bound on total displacement over the episode (t>0 frames).
        # Keep it small but sufficient for 13f/1chunk debug runs and beyond.
        margin = float(move_step * max(1, (num_frames - 1)) + 0.25)
        world_bounds_xz = (min_x - margin, max_x + margin, min_z - margin, max_z + margin)

    K = make_intrinsic(width=width, height=height, fov_deg=fov_deg)
    rays_cam = precompute_camera_rays_cam(width, height, K)

    state = BallWorldState()
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

    if hold_action_frames <= 0:
        raise ValueError("--hold_action_frames must be a positive integer")

    # Keep actions constant within each block of `hold_action_frames` frames to align with
    # latent steps (1 latent ~= 4 frames). This reduces supervision noise since training/eval
    # typically sample action at 0,4,8,...
    cur_move_action = ""
    cur_view_action = ""

    for t in range(num_frames):
        if t == 0:
            cur_move_action = ""
            cur_view_action = ""
        else:
            # Update actions only on block boundaries (e.g., t=4,8,12,... for hold_action_frames=4)
            # to align with latent steps (1 latent ~= 4 frames).
            if (t % hold_action_frames) == 0:
                # Occasionally change the macro actions to avoid trivial straight lines
                if macro_period > 0 and (t % macro_period) == 0:
                    macro_move, macro_view = _sample_episode_macro_actions(rng)

                # Allow fixing move/view independently:
                # - fixed_view_action="" + fixed_move_action=None => "no view rotation + random move"
                # - fixed_move_action="W" + fixed_view_action=None => "fixed move + random view"
                if fixed_move_action_fn is not None:
                    cur_move_action = str(fixed_move_action_fn(t)) or ""
                elif fixed_move_action is not None:
                    cur_move_action = fixed_move_action or ""
                else:
                    cur_move_action = _micro_action(
                        rng,
                        macro_move,
                        empty_prob=move_empty_prob,
                        macro_prob=move_macro_prob,
                        alt_prob=move_alt_prob,
                        alts=move_alts,
                    )

                if fixed_view_action is not None:
                    cur_view_action = fixed_view_action or ""
                else:
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
    p.add_argument(
        "--hold_action_frames",
        type=int,
        default=4,
        help="Hold move/view action constant for this many frames (default: 4). "
        "Use 4 to align with 1 latent ~= 4 frames.",
    )
    p.add_argument(
        "--fixed_move_action_mode",
        type=str,
        default="single",
        choices=["single", "4dir", "4dirback"],
        help=(
            "How to apply --fixed_move_action. "
            "'single': all samples share the same fixed move action (legacy behavior). "
            "'4dir': ignore --fixed_move_action and generate 4 samples with fixed move actions "
            "[W,S,A,D] (front/back/left/right), keeping the rest identical. "
            "'4dirback': ignore --fixed_move_action and generate 4 samples; each sample moves in a fixed direction "
            "for the first half, then moves in the opposite direction for the second half (optionally inserting "
            "one neutral block if the number of action blocks is odd) to approximately return to the origin."
        ),
    )
    p.add_argument(
        "--fixed_move_action",
        type=str,
        default=None,
        help="If set, force a fixed move_action for all t>0 frames (e.g. 'W', 'A', 'S', 'D', 'WA'). "
        "This enables a simple/debug mode (reduced randomness).",
    )
    p.add_argument(
        "--fixed_view_action",
        type=str,
        default=None,
        help="If set, force a fixed view_action for all t>0 frames (e.g. '', 'LR', 'LL', 'LU', 'LD'). "
        "Use '' to keep camera orientation fixed.",
    )
    args = p.parse_args(argv)

    out_root = Path(args.out_root).expanduser().resolve()
    split_dir = out_root / args.split
    _ensure_dir(split_dir)

    manifest: list[dict] = []

    def _make_4dirback_fn(base_dir: str, *, num_frames: int,
                          hold_action_frames: int) -> Callable[[int], str]:
        opposite = {"W": "S", "S": "W", "A": "D", "D": "A"}
        if base_dir not in opposite:
            raise ValueError(
                f"4dirback expects base_dir in {list(opposite.keys())}, got: {base_dir}"
            )
        opp_dir = opposite[base_dir]

        # Actions are updated only at t % hold_action_frames == 0 and t>0.
        # Count how many such update blocks exist across the episode.
        total_blocks = max(0, (num_frames - 1) // hold_action_frames)
        forward_blocks = total_blocks // 2
        middle_block = forward_blocks + 1 if (total_blocks % 2 == 1) else None

        def _fn(t: int) -> str:
            if t <= 0:
                return ""
            block_idx = t // hold_action_frames  # 1..total_blocks
            if block_idx <= forward_blocks:
                return base_dir
            if middle_block is not None and block_idx == middle_block:
                return ""  # neutral to balance when total_blocks is odd
            return opp_dir

        return _fn

    for i in range(args.num_samples):
        fixed_move_action = args.fixed_move_action
        fixed_move_action_fn: Callable[[int], str] | None = None
        if args.fixed_move_action_mode == "4dir":
            # Deterministic 4-direction dataset: front/back/left/right -> W/S/A/D in world axes.
            fixed_move_action = ["W", "S", "A", "D"][i % 4]
        elif args.fixed_move_action_mode == "4dirback":
            base_dir = ["W", "S", "A", "D"][i % 4]
            fixed_move_action = None
            fixed_move_action_fn = _make_4dirback_fn(
                base_dir,
                num_frames=args.num_frames,
                hold_action_frames=args.hold_action_frames,
            )

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
            hold_action_frames=args.hold_action_frames,
            fixed_move_action=fixed_move_action,
            fixed_view_action=args.fixed_view_action,
            fixed_move_action_fn=fixed_move_action_fn,
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


