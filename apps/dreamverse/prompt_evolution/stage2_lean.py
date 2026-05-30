"""Lean VIDEO-only Stage-2 scorer (no audio, no heavy VBench package).

Combines two self-contained, independently-validated pieces:
  - Shao's `boundary.video` (reasoning-free DINO seam-reset detector + RAFT/LPIPS context)
    -> the seam-artifact PENALTY. This is the key signal coffee_argue lacked.
  - our `video_scorer` (transformers-CLIP, numpy/scipy) -> quality / motion / and
    `text_alignment` = PROMPT CONSISTENCY, all dependency-light.

  combined = video_scorer.video_score - 0.15 * min(boundary.video severity_per_seam, cap)

Only `boundary.video` is requested from the fastvideo.eval registry, so the vbench.*
metrics (which need the heavy `vbench` package: ViCLIP / third_party RAFT) are never
imported. The evaluator is created once and reused. Higher combined = better.
"""
import os
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DEVICE = os.environ.get("METRIC_DEVICE", "cuda:0")
SEVERITY_CAP = float(os.environ.get("SEV_CAP", "8.0"))
SEAM_WEIGHT = float(os.environ.get("SEAM_WEIGHT", "0.15"))

_EV = None


def _evaluator() -> Any:
    global _EV
    if _EV is None:
        from fastvideo.eval import create_evaluator
        _EV = create_evaluator(metrics=["boundary.video"], device=DEVICE)
    return _EV


def score(video_path: str, rinfo: dict, segments: list, prompt: str):
    """Return (combined_score, metrics_dict, artifacts_dict)."""
    from fastvideo.eval import samples_from
    seams = rinfo.get("seam_frames")
    res = _evaluator().evaluate(samples=samples_from(
        video=video_path, text_prompt=prompt, fps=24.0, auxiliary_info={"seam_frames": seams} if seams else None))[0]
    bv = res.get("boundary.video")
    bvd = (getattr(bv, "details", {}) or {}) if bv else {}
    sev = min(float(bvd.get("severity_per_seam") or 0.0), SEVERITY_CAP)

    import video_scorer as vs
    q = vs.score_rollout(video_path, segment_prompts=segments, segment_boundaries=rinfo.get("segment_frame_counts"))
    quality = q.get("video_score", 0.0)
    combined = round(quality - SEAM_WEIGHT * sev, 4)

    metrics = {k: v for k, v in q.items() if k not in ("video_score", "clip_used", "num_frames", "num_segments")}
    metrics.update({
        "quality_mean": round(quality, 4),
        "prompt_consistency": q.get("text_alignment"),  # CLIP text<->frames (alias)
        "video_artifact_rate": getattr(bv, "score", None) if bv else None,
        "video_severity_per_seam": bvd.get("severity_per_seam"),
        "n_video_artifacts": bvd.get("n_artifacts"),
        "n_seams": bvd.get("n_seams"),
    })
    b = bvd.get("boundaries") or []
    artifacts = {
        "video_artifacts": "none" if not b else "; ".join(f"{x['time_s']}s z={x['z']}({x['signal']})" for x in b)
    }
    return combined, metrics, artifacts
