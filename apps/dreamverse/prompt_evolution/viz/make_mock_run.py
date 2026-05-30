"""
Build a mock evolution run so the visualization is demoable before real renders exist.

Produces  viz/runs/<run_id>/evolution.json  and a  videos/  dir of symlinks to
sample clips in the repo. The schema matches exactly what the real evolve loop
will emit (combined_score + video_scorer dims + stage-1 seam), so swapping in real
data is a drop-in replacement.

Usage:  python make_mock_run.py [run_id]
"""

import contextlib
import json
import math
import os
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))

# Sample clips in the repo used as stand-in rollout videos.
SAMPLE_VIDEOS = [
    "assets/videos/robot_pouring.mp4",
    "assets/motorcycle.mp4",
    "assets/8steps/mochi-demo.mp4",
    "outputs/longcat_self_forcing_local/validation_step_0_inference_steps_4_video_0.mp4",
    "outputs/longcat_self_forcing_local/validation_step_0_inference_steps_4_video_1.mp4",
    "outputs/longcat_self_forcing_local/validation_step_0_inference_steps_4_video_2.mp4",
]

DIMS = [
    "text_alignment", "segment_consistency", "dynamic_degree", "motion_smoothness", "temporal_flicker", "sharpness",
    "colorfulness", "seam_continuity"
]

MUTATIONS = [
    "seed prompt (baseline policy blocks)",
    "tightened handoff policy: end each segment on a named stable subject",
    "added explicit anti-static rule + one motion beat per segment",
    "compressed house_style; sharper camera-move guidance at segment 4",
    "stronger audio carry-forward; dialogue placed before final sentence",
    "merged best handoff + anti-static blocks; trimmed redundancy",
    "added identity-anchor reminder to reduce subject drift across pivot",
    "shortened policy (latency win) while keeping seam + dynamic rules",
]


def _score(gen: int, k: int, n_gen: int) -> dict:
    """Plausible per-dim scores that trend up across generations with spread."""
    base = 0.55 + 0.30 * (gen / max(1, n_gen - 1))  # 0.55 -> 0.85
    out = {}
    for i, d in enumerate(DIMS):
        jitter = 0.12 * math.sin(gen * 1.7 + k * 2.3 + i * 0.9)
        bonus = 0.05 if d in ("dynamic_degree", "seam_continuity") and gen >= 2 else 0.0
        out[d] = round(min(0.99, max(0.05, base + jitter + bonus - 0.04 * k)), 4)
    return out


def build(run_id: str, n_gen: int = 5, per_gen: int = 3) -> str:
    run_dir = os.path.join(HERE, "runs", run_id)
    vids_dir = os.path.join(run_dir, "videos")
    os.makedirs(vids_dir, exist_ok=True)

    generations: list[Any] = []
    cand_counter = 0
    prev_ids: list[str] = []
    for gen in range(n_gen):
        cands: list[Any] = []
        for k in range(per_gen):
            dims = _score(gen, k, n_gen)
            combined = round(sum(dims.values()) / len(dims), 4)
            src = os.path.join(REPO_ROOT, SAMPLE_VIDEOS[cand_counter % len(SAMPLE_VIDEOS)])
            link = os.path.join(vids_dir, f"cand_{cand_counter}.mp4")
            if os.path.exists(src) and not os.path.exists(link):
                with contextlib.suppress(OSError):
                    os.symlink(src, link)
            cands.append({
                "id":
                f"candidate_{cand_counter}",
                "generation":
                gen,
                "parent_id": (prev_ids[k % len(prev_ids)] if prev_ids else None),
                "island":
                cand_counter % 3,
                "combined_score":
                combined,
                "video":
                f"videos/cand_{cand_counter}.mp4",
                "metrics":
                dims,
                "prompt_length":
                7153 - gen * 180 + k * 60,
                "mutation_summary":
                MUTATIONS[cand_counter % len(MUTATIONS)],
                "prompt_excerpt":
                "<house_style> ... one readable beat per segment; "
                "end on a named stable subject; one motion beat ...",
            })
            cand_counter += 1
        cands.sort(key=lambda c: -c["combined_score"])
        prev_ids = [c["id"] for c in cands]
        generations.append({"generation": gen, "candidates": cands})

    data = {
        "run_id": run_id,
        "target": "rewrite_new_rollout system prompt",
        "evolver_model": "gpt-5.1",
        "task_model": "gpt-oss-120b",
        "feature_dimensions": ["prompt_length", "static_rate"],
        "note": "MOCK DATA — sample clips stand in for rendered rollouts.",
        "generations": generations,
    }
    with open(os.path.join(run_dir, "evolution.json"), "w") as f:
        json.dump(data, f, indent=2)
    return run_dir


if __name__ == "__main__":
    rid = sys.argv[1] if len(sys.argv) > 1 else "mock"
    d = build(rid)
    print(f"wrote {d}/evolution.json")
