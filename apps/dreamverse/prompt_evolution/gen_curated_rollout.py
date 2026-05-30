"""Render a 30s rollout from 6 CURATED segment prompts, enhancement OFF (verbatim,
deterministic — the server does NOT rewrite). Mirrors dreamverse_setup/gen_lego_rollout.py
but takes the prompts from a json file instead of a preset.

Usage: python gen_curated_rollout.py <segment_prompts.json> <out_dir>
Writes <out_dir>/rollout_final.mp4 + <out_dir>/manifest.json (with empirical seam_frames).
"""
import asyncio
import json
import os
import subprocess
import sys
import time

import websockets

SEGJSON, OUTDIR = sys.argv[1], sys.argv[2]
HOST = os.environ.get("DV_WS_HOST", "hpc-rack-2-3")
PORT = os.environ.get("DV_WS_PORT", "8009")
URL = f"ws://{HOST}:{PORT}/ws"
SEGDIR = os.path.join(OUTDIR, "segments")
FFMPEG = os.environ.get("FASTVIDEO_FFMPEG_BIN", "/home/hal-shao/miniforge3/bin/ffmpeg")
FFPROBE = os.environ.get("FASTVIDEO_FFPROBE_BIN", "/home/hal-shao/miniforge3/bin/ffprobe")
PROMPTS = [p.strip() for p in json.load(open(SEGJSON))["segment_prompts"] if p and p.strip()]


async def run():
    os.makedirs(SEGDIR, exist_ok=True)
    init = {"type": "session_init_v2", "curated_prompts": PROMPTS, "enhancement_enabled": False,
            "auto_extension_enabled": False, "loop_generation_enabled": False}
    total = len(PROMPTS)
    buffers, seg_files, done = {}, {}, set()
    cur = None
    t0 = time.time()
    print(f"[client] connecting {URL} curated_segments={total}", flush=True)
    async with websockets.connect(URL, max_size=None, ping_interval=None, open_timeout=30) as ws:
        await ws.send(json.dumps(init))
        print("[client] session_init_v2 (curated, enhancement OFF) sent", flush=True)
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1200)
            except asyncio.TimeoutError:
                print("[client] recv timeout (1200s) - aborting", flush=True)
                break
            if isinstance(msg, (bytes, bytearray)):
                if cur is not None:
                    buffers.setdefault(cur, bytearray()).extend(msg)
                continue
            evt = json.loads(msg)
            t = evt.get("type")
            if t == "ltx2_stream_start":
                total = evt.get("total_segments", total)
                print(f"[client] stream_start total_segments={total}", flush=True)
            elif t == "ltx2_segment_start":
                cur = evt.get("segment_idx")
                buffers.setdefault(cur, bytearray())
                print(f"[client] segment_start idx={cur} t=+{time.time()-t0:.0f}s", flush=True)
            elif t == "media_init":
                cur = evt.get("segment_idx", cur)
                buffers.setdefault(cur, bytearray())
            elif t == "media_segment_complete":
                idx = evt.get("segment_idx", cur)
                buf = buffers.get(idx, bytearray())
                path = os.path.join(SEGDIR, f"seg_{idx:02d}.mp4")
                open(path, "wb").write(buf)
                seg_files[idx] = path
                print(f"[client] segment {idx} saved {len(buf)/1e6:.2f}MB", flush=True)
            elif t == "ltx2_segment_complete":
                done.add(evt.get("segment_idx"))
                print(f"[client] segment_complete ({len(done)}/{total}) t=+{time.time()-t0:.0f}s", flush=True)
                if total and len(done) >= total:
                    try:
                        await ws.send(json.dumps({"type": "leave"}))
                    except Exception:
                        pass
                    break
            elif t == "ltx2_stream_complete":
                print("[client] stream_complete", flush=True)
                break
            elif t in ("queue_status", "gpu_assigned", "step_complete"):
                pass
            else:
                print(f"[client] evt={t}", flush=True)
    print(f"[client] received {len(seg_files)} segments in {time.time()-t0:.0f}s", flush=True)
    return seg_files


def ffprobe_frames(path):
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames,duration", "-of", "json", path],
        capture_output=True, text=True)
    try:
        s = json.loads(out.stdout)["streams"][0]
        return int(s.get("nb_read_frames", 0)), float(s.get("duration", 0) or 0)
    except Exception:
        return None, None


def finalize(seg_files):
    idxs = sorted(seg_files)
    counts, durs = {}, {}
    for i in idxs:
        counts[i], durs[i] = ffprobe_frames(seg_files[i])
    listfile = os.path.join(SEGDIR, "concat.txt")
    open(listfile, "w").write("".join(f"file '{seg_files[i]}'\n" for i in idxs))
    final = os.path.join(OUTDIR, "rollout_final.mp4")
    r = subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-fflags", "+genpts", "-f", "concat",
                        "-safe", "0", "-i", listfile, "-c", "copy", final], capture_output=True, text=True)
    if r.returncode != 0:
        print("[client] -c copy concat failed; re-encoding (libx264/aac)", flush=True)
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", listfile,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", final], check=True)
    seams, cum = [], 0
    for i in idxs[:-1]:
        cum += (counts[i] or 0)
        seams.append(cum)
    fn, fdur = ffprobe_frames(final)
    manifest = {
        "segment_prompts": PROMPTS, "segments": {str(k): v for k, v in seg_files.items()},
        "per_segment_frames": {str(k): counts[k] for k in idxs},
        "per_segment_duration_s": {str(k): durs[k] for k in idxs},
        "seam_frames": seams, "seam_times_s": [round(s / 24.0, 3) for s in seams],
        "final": final, "final_frames": fn, "final_duration_s": fdur, "fps": 24,
    }
    json.dump(manifest, open(os.path.join(OUTDIR, "manifest.json"), "w"), indent=2)
    print(f"[client] per-segment frames: {counts}", flush=True)
    print(f"[client] SEAM frames: {seams} times={[round(s/24.0,3) for s in seams]}", flush=True)
    print(f"[client] FINAL {final} frames={fn} dur={fdur}s", flush=True)
    return final, seams


if __name__ == "__main__":
    files = asyncio.run(run())
    if not files:
        print("[client] NO SEGMENTS", flush=True)
        raise SystemExit(1)
    finalize(files)
    print("[client] DONE", flush=True)
