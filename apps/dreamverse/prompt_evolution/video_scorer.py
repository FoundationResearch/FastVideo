"""
Lightweight video quality scorer for the evolve loop (VBench-style, no VBench install).

Implements the agreed "quality + dynamic" recipe, chosen so that the deliberate
segment-4 scene pivot in a 30s rollout is NOT punished and static "wallpaper" video
is NOT rewarded (the Goodhart trap of naive whole-clip consistency metrics):

  - text_alignment        CLIP similarity between each segment's prompt and its frames
  - segment_consistency   CLIP frame-frame similarity WITHIN a segment only (not across)
  - dynamic_degree        motion magnitude (frame diff) — rewarded UP to a target band
  - motion_smoothness     low jerk in the motion signal
  - temporal_flicker      low within-segment brightness jitter
  - sharpness             Laplacian variance (imaging quality proxy)
  - colorfulness          Hasler-Susstrunk colorfulness (aesthetic proxy)

CLIP (transformers, openai/clip-vit-base-patch32) is optional and lazy-loaded; if it
or its weights are unavailable, the CLIP-based dims are dropped and the remaining
dims are renormalized. All dims are in [0,1]; `video_score` is their weighted mean.

Weights and normalization constants are heuristic and meant to be tuned / moved to
config.yaml later. Full VBench can replace/augment this behind the same interface.

Usage:
    python video_scorer.py <rollout.mp4> [segment_prompts.json] [segment_boundaries.json]
"""

import json
import os
import sys
from typing import Any

import numpy as np

WEIGHTS = {
    "text_alignment": 0.30,  # CLIP (dropped if unavailable)
    "segment_consistency": 0.15,  # CLIP (dropped if unavailable)
    "dynamic_degree": 0.15,
    "motion_smoothness": 0.10,
    "temporal_flicker": 0.10,
    "sharpness": 0.10,
    "colorfulness": 0.10,
}
_CLIP_DIMS = {"text_alignment", "segment_consistency"}

# Normalization targets (heuristic).
DYNAMIC_TARGET = 0.06  # mean abs grayscale frame-diff (fraction of 255) for "good" motion
SHARPNESS_TARGET = 400.0  # Laplacian variance for "sharp"
COLOR_TARGET = 60.0  # Hasler-Susstrunk colorfulness for "vivid"
FLICKER_SCALE = 12.0  # per-frame mean-brightness std (0-255) mapping to flicker


# --- frame IO ----------------------------------------------------------------
def _resize_long(frame: np.ndarray, long_side: int) -> np.ndarray:
    from PIL import Image
    h, w = frame.shape[:2]
    if max(h, w) <= long_side:
        return frame
    scale = long_side / max(h, w)
    im = Image.fromarray(frame).resize((max(1, int(w * scale)), max(1, int(h * scale))))
    return np.asarray(im)


def load_frames(mp4_path: str, max_frames: int = 240, long_side: int = 320) -> np.ndarray:
    """Decode up to max_frames RGB uint8 frames via imageio-ffmpeg (no libGL/cv2 needed),
    uniformly subsampling and downscaling to bound memory."""
    import imageio.v2 as imageio
    reader = imageio.get_reader(mp4_path, "ffmpeg")
    try:
        total = reader.count_frames()
    except Exception:
        total = None
    step = max(1, total // max_frames) if (total and total > max_frames) else 1
    frames = []
    for i, fr in enumerate(reader):
        if i % step == 0:
            frames.append(_resize_long(np.asarray(fr)[..., :3], long_side))
    reader.close()
    if not frames:
        raise RuntimeError(f"no frames decoded from {mp4_path}")
    return np.stack(frames)


def _segments(n_frames: int, boundaries):
    """Yield (start, end) frame index ranges per segment."""
    if boundaries:
        idx = 0
        for c in boundaries:
            yield idx, min(idx + c, n_frames)
            idx += c
    else:  # split evenly into 6
        per = max(1, n_frames // 6)
        for s in range(0, n_frames, per):
            yield s, min(s + per, n_frames)


# --- non-CLIP metrics (numpy / cv2) -----------------------------------------
def _gray(frames: np.ndarray) -> np.ndarray:
    return frames.astype(np.float32).mean(axis=3)  # NxHxW


def _dynamic_and_smoothness(gray: np.ndarray):
    if len(gray) < 2:
        return 0.0, 1.0
    diffs = np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2)) / 255.0  # per-frame motion
    dynamic = float(np.clip(diffs.mean() / DYNAMIC_TARGET, 0.0, 1.0))
    if len(diffs) < 2:
        return dynamic, 1.0
    jerk = np.abs(np.diff(diffs)).mean()
    smoothness = float(np.clip(1.0 - jerk / DYNAMIC_TARGET, 0.0, 1.0))
    return dynamic, smoothness


def _flicker(gray: np.ndarray) -> float:
    if len(gray) < 2:
        return 1.0
    brightness = gray.mean(axis=(1, 2))
    return float(np.clip(1.0 - brightness.std() / FLICKER_SCALE, 0.0, 1.0))


def _sharpness(frames: np.ndarray) -> float:
    from scipy.ndimage import laplace
    sample = frames[::max(1, len(frames) // 8)]
    vals = []
    for f in sample:
        g = f.astype(np.float32).mean(axis=2)
        vals.append(laplace(g).var())
    return float(np.clip(np.mean(vals) / SHARPNESS_TARGET, 0.0, 1.0))


def _colorfulness(frames: np.ndarray) -> float:
    sample = frames[::max(1, len(frames) // 8)].astype(np.float32)
    r, g, b = sample[..., 0], sample[..., 1], sample[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    std = np.sqrt(rg.std()**2 + yb.std()**2)
    mean = np.sqrt(rg.mean()**2 + yb.mean()**2)
    return float(np.clip((std + 0.3 * mean) / COLOR_TARGET, 0.0, 1.0))


# --- CLIP metrics (optional) -------------------------------------------------
_CLIP: Any = None


def _get_clip() -> Any:
    global _CLIP
    if os.environ.get("SCORER_NO_CLIP") == "1":
        return None
    if _CLIP == "unavailable":
        return None
    if _CLIP is None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            name = os.environ.get("CLIP_MODEL", "openai/clip-vit-base-patch32")
            model = CLIPModel.from_pretrained(name).eval()
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(dev)
            proc = CLIPProcessor.from_pretrained(name)
            _CLIP = (model, proc, dev, torch)
        except Exception as e:
            print(f"[scorer] CLIP unavailable ({e}); dropping CLIP dims", file=sys.stderr)
            _CLIP = "unavailable"
            return None
    return _CLIP


def _clip_image_embeds(frames: np.ndarray):
    clip = _get_clip()
    if clip is None:
        return None
    model, proc, dev, torch = clip
    sample = frames[::max(1, len(frames) // 6)]
    with torch.no_grad():
        inp = proc(images=list(sample), return_tensors="pt").to(dev)
        emb = model.get_image_features(**inp)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb  # [k, d] on device


def _clip_text_embed(text: str):
    clip = _get_clip()
    if clip is None:
        return None
    model, proc, dev, torch = clip
    with torch.no_grad():
        inp = proc(text=[text[:300]], return_tensors="pt", padding=True, truncation=True).to(dev)
        emb = model.get_text_features(**inp)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0]


# --- main scoring ------------------------------------------------------------
def score_rollout(mp4_path: str, segment_prompts=None, segment_boundaries=None) -> dict:
    frames = load_frames(mp4_path)
    n = len(frames)
    segs = list(_segments(n, segment_boundaries))

    # whole-clip non-CLIP dims
    gray = _gray(frames)
    dynamic, smoothness = _dynamic_and_smoothness(gray)
    dims = {
        "dynamic_degree": dynamic,
        "motion_smoothness": smoothness,
        "sharpness": _sharpness(frames),
        "colorfulness": _colorfulness(frames),
    }
    # flicker computed per-segment then averaged (within-segment jitter only)
    dims["temporal_flicker"] = float(np.mean([_flicker(gray[s:e]) for s, e in segs]))

    # CLIP dims (optional)
    clip_ok = _get_clip() is not None
    if clip_ok:
        seg_consist, seg_align = [], []
        for i, (s, e) in enumerate(segs):
            emb = _clip_image_embeds(frames[s:e])
            if emb is None or len(emb) < 1:
                continue
            if len(emb) >= 2:
                sim = (emb @ emb.T)
                k = sim.shape[0]
                off = (sim.sum() - k) / (k * (k - 1))  # mean off-diagonal
                seg_consist.append(float(off))
            if segment_prompts and i < len(segment_prompts):
                t = _clip_text_embed(segment_prompts[i])
                if t is not None:
                    seg_align.append(float((emb @ t).mean()))
        if seg_consist:
            dims["segment_consistency"] = float(np.clip(np.mean(seg_consist), 0.0, 1.0))
        if seg_align:
            # CLIP cos sim ~0.2-0.35 typical; rescale to [0,1] around that band
            dims["text_alignment"] = float(np.clip((np.mean(seg_align) - 0.15) / 0.20, 0.0, 1.0))

    # weighted combine over available dims (renormalize if CLIP dims missing)
    active = {k: v for k, v in WEIGHTS.items() if k in dims}
    wsum = sum(active.values())
    video_score = float(sum(dims[k] * w for k, w in active.items()) / wsum) if wsum else 0.0

    return {
        "video_score": round(video_score, 4),
        "clip_used": clip_ok,
        "num_frames": n,
        "num_segments": len(segs),
        **{
            k: round(v, 4)
            for k, v in dims.items()
        },
    }


if __name__ == "__main__":
    mp4 = sys.argv[1] if len(sys.argv) > 1 else None
    if not mp4:
        print("usage: python video_scorer.py <rollout.mp4> [prompts.json] [boundaries.json]")
        sys.exit(1)

    def _load(path: str) -> Any:
        with open(path) as f:
            return json.load(f)

    prompts = _load(sys.argv[2]) if len(sys.argv) > 2 else None
    bounds = _load(sys.argv[3]) if len(sys.argv) > 3 else None
    print(json.dumps(score_rollout(mp4, prompts, bounds), indent=2))
