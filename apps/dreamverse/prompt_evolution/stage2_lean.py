"""Lean Stage-2 scorer for the evolution loop (video + audio, no heavy VBench package).

Pieces (all self-contained, no `vbench` pip pkg):
  - video_scorer (transformers-CLIP, numpy) -> quality / motion / `text_alignment`
    = PROMPT CONSISTENCY.
  - boundary.video (Shao) -> seam content-reset PENALTY (reliable).
  - boundary.audio (Shao) -> seam audio-discontinuity PENALTY. CAVEAT: LTX2 audio is
    near-silent so its robust-z saturates and this is mostly a ~constant offset (Shao:
    "不太好用"); kept clipped + low-weight so it can't dominate.
  - audio.noise (Shao) -> bounded tonal "whirring" noise PENALTY (the usable audio one).

  combined = quality
           - 0.15 * min(boundary.video sev, VID_CAP)
           - 0.10 * min(boundary.audio sev, AUD_CAP)
           - 0.05 * audio.noise

Only these four metrics are requested from the registry. Evaluator built once & reused.
Higher combined = better. Weights/caps are env-tunable.
"""
import os
import subprocess
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DEVICE = os.environ.get("METRIC_DEVICE", "cuda:0")
FFMPEG = os.environ.get("FASTVIDEO_FFMPEG_BIN", "ffmpeg")
VID_CAP = float(os.environ.get("VID_SEV_CAP", "8.0"))
AUD_CAP = float(os.environ.get("AUD_SEV_CAP", "6.0"))
W_VID = float(os.environ.get("W_VID_SEAM", "0.15"))
W_NOISE = float(os.environ.get("W_NOISE", "0.05"))
# boundary.audio removed: its robust-z saturates on LTX2's near-silent audio and
# produced high-variance outliers (e.g. one prompt at -0.85) that drowned the signal.
METRICS = ["boundary.video", "audio.noise"]

_EV = None


def _evaluator() -> Any:
    global _EV
    if _EV is None:
        from fastvideo.eval import create_evaluator
        _EV = create_evaluator(metrics=METRICS, device=DEVICE)
    return _EV


def _extract_wav(video_path: str) -> str | None:
    wav = video_path + ".eval.wav"
    try:
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", video_path, "-ac", "1", "-ar", "48000", wav],
                       check=True)
        return wav
    except Exception:
        return None


def score(video_path: str, rinfo: dict, segments: list, prompt: str):
    """Return (combined_score, metrics_dict, artifacts_dict)."""
    from fastvideo.eval import samples_from
    seams = rinfo.get("seam_frames")
    wav = _extract_wav(video_path)
    sample_kwargs = dict(video=video_path,
                         text_prompt=prompt,
                         fps=24.0,
                         auxiliary_info={"seam_frames": seams} if seams else None)
    if wav:
        sample_kwargs["audio"] = wav
    res = _evaluator().evaluate(samples=samples_from(**sample_kwargs))[0]
    if wav:
        import contextlib
        with contextlib.suppress(OSError):
            os.remove(wav)

    def sc(name: str) -> Any:
        r = res.get(name)
        return getattr(r, "score", None) if r else None

    def det(name: str) -> dict:
        r = res.get(name)
        return (getattr(r, "details", {}) or {}) if r else {}

    bvd = det("boundary.video")
    vid_sev = min(float(bvd.get("severity_per_seam") or 0.0), VID_CAP)
    noise = float(sc("audio.noise") or 0.0)

    import video_scorer as vs
    q = vs.score_rollout(video_path, segment_prompts=segments, segment_boundaries=rinfo.get("segment_frame_counts"))
    quality = q.get("video_score", 0.0)
    combined = round(quality - W_VID * vid_sev - W_NOISE * noise, 4)

    metrics = {k: v for k, v in q.items() if k not in ("video_score", "clip_used", "num_frames", "num_segments")}
    metrics.update({
        "quality_mean": round(quality, 4),
        "prompt_consistency": q.get("text_alignment"),
        "video_artifact_rate": sc("boundary.video"),
        "video_severity_per_seam": bvd.get("severity_per_seam"),
        "audio_noise_score": sc("audio.noise"),
        "squim_pesq": det("audio.noise").get("squim_pesq"),
        "n_seams": bvd.get("n_seams"),
    })

    def _fmt(d: dict) -> str:
        b = d.get("boundaries") or []
        return "none" if not b else "; ".join(f"{x['time_s']}s z={x['z']}({x['signal']})" for x in b)

    artifacts = {
        "video_artifacts": _fmt(bvd),
        "noise": f"score={sc('audio.noise')} pesq={det('audio.noise').get('squim_pesq')}"
    }
    return combined, metrics, artifacts
