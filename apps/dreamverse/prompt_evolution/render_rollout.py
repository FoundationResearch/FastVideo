"""
Offline renderer: 6 segment prompts -> one continuous 30s mp4 (video-only).

Reuses the production dreamverse generator (VideoGenerationWorker.generate_step),
which internally chains each segment on the previous segment's last 9 video frames
+ 49 audio frames. We collect the (head-trimmed) frames of all 6 segments and mux
to a standard mp4 with ffmpeg.

This is the shared foundation for Stage-2 VBench scoring and the evolution
visualization. Audio muxing is a TODO (needed for WER eval, not for VBench).

Usage:
    source env.local.sh
    python render_rollout.py                      # render a built-in demo rollout
    python render_rollout.py prompts.json out.mp4 # render given 6 prompts
"""

import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DREAMVERSE_ROOT = os.environ.get("DREAMVERSE_ROOT", os.path.dirname(HERE))
# Repo root (.../FastVideo) must lead sys.path so `import fastvideo` resolves to the
# workspace source (which has fastvideo.api), not an older site-packages build.
REPO_ROOT = os.path.dirname(os.path.dirname(DREAMVERSE_ROOT))
for _p in (REPO_ROOT, DREAMVERSE_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
FFMPEG_BIN = os.environ.get("FASTVIDEO_FFMPEG_BIN", "ffmpeg")


def render_rollout(prompts: list, out_mp4: str, *, gpu_id: int = 0, fps: int = 24) -> dict:
    """Render `prompts` (6 segment strings) into `out_mp4`. Returns timing/shape info."""
    if DREAMVERSE_ROOT not in sys.path:
        sys.path.insert(0, DREAMVERSE_ROOT)
    from dreamverse.video_generation import VideoGenerationWorker  # type: ignore

    import time
    t0 = time.perf_counter()
    worker = VideoGenerationWorker(gpu_id=gpu_id)
    worker.initialize(None)  # default MODEL_CONFIG (FastVideo/LTX2-Distilled-Diffusers)
    t_init = time.perf_counter() - t0

    frames: list = []
    seg_timings = []
    for idx, prompt in enumerate(prompts, start=1):
        ts = time.perf_counter()
        step = worker.generate_step(
            prompt=prompt,
            segment_idx=idx,
            image_path=None,
            reset_conditioning=(idx == 1),
        )
        segf = step.frames
        if getattr(step, "head_trim_frames", 0):
            segf = segf[step.head_trim_frames:]
        frames.extend(segf)
        seg_timings.append(round(time.perf_counter() - ts, 2))

    worker.shutdown()

    os.makedirs(os.path.dirname(os.path.abspath(out_mp4)), exist_ok=True)
    _write_mp4(frames, out_mp4, fps)

    return {
        "out_mp4": out_mp4,
        "num_frames": len(frames),
        "duration_s": round(len(frames) / fps, 2),
        "init_s": round(t_init, 2),
        "segment_s": seg_timings,
    }


def _write_mp4(frames: list, out: str, fps: int) -> None:
    if not frames:
        raise RuntimeError("no frames to write")
    h, w = frames[0].shape[:2]
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        out,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for fr in frames:
        proc.stdin.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited {rc}")


_DEMO_PROMPTS = [
    "A cramped sunlit kitchen at morning. Two roommates, Mia in a green hoodie and Jon in a "
    "rumpled t-shirt, face a single steaming coffee mug on the counter. Mia reaches for it. "
    "Jon's hand lands on the handle first. They freeze, eyes locked. Soft fridge hum.",
    "Close on the mug between their hands. Mia narrows her eyes. \"That's mine.\" Jon smirks. "
    "Steam curls upward. The handle stays gripped by both.",
    "Mia tugs the mug an inch. Coffee ripples but doesn't spill. Jon leans in, not letting go. "
    "\"Finders keepers,\" he says quietly. The counter light glints on the ceramic.",
    "Wide shot: the kitchen door swings open and a sleepy third roommate shuffles in, hair "
    "askew, holding an empty mug. Both freeze and look over. The contested mug sits still "
    "between them on the counter.",
    "The newcomer pours from a fresh pot on the warmer, oblivious. Mia and Jon exchange a "
    "defeated glance. Jon slowly slides the mug toward Mia. \"Take it.\"",
    "Mia lifts the mug with a small grin and sips. Jon slumps against the counter, arms "
    "folded, half-smiling. Warm morning light, gentle fridge hum, a calm held frame.",
]

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        with open(sys.argv[1]) as _f:
            prompts = json.load(_f)
        out = sys.argv[2]
    else:
        prompts = _DEMO_PROMPTS
        out = os.path.join(HERE, "renders", "demo_rollout.mp4")
    info = render_rollout(prompts, out)
    print(json.dumps(info, indent=2))
