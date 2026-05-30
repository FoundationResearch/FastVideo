"""Per-GPU batch worker for the parallel evolution loop.

Launched as a subprocess with CUDA_VISIBLE_DEVICES pinned to ONE physical GPU. It
processes a batch of candidates assigned to that GPU in two internal phases so the
render worker and the metric models never fight for memory on the same card:

  1. RENDER phase  — load the LTX2 worker once, render every candidate's 6 segments
     to its mp4 (collecting seam_frames), then CLOSE the worker to free GPU memory.
  2. SCORE phase   — load stage2_lean's metric evaluator and score every rendered mp4.

I/O: a single work JSON (list of items). Each item in:
  {"id","segments":[6],"eval_idea","out_mp4"}
is updated in place with:
  {"combined","metrics","seam_frames","error"}

Usage (env: source env.local.sh; LD_LIBRARY_PATH for cuda):
  CUDA_VISIBLE_DEVICES=2 METRIC_DEVICE=cuda:0 python gpu_worker.py --work batch.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    args = ap.parse_args()
    with open(args.work) as f:
        items = json.load(f)

    from render_rollout import RolloutRenderer  # noqa: E402

    # --- phase 1: render everything on this GPU, then free the worker ---
    renderer = RolloutRenderer(gpu_id=0)  # gpu_id 0 = the single CUDA_VISIBLE_DEVICES card
    for it in items:
        try:
            it["rinfo"] = renderer.render(it["segments"], it["out_mp4"])
            it["seam_frames"] = it["rinfo"].get("seam_frames")
        except Exception as e:
            it["error"] = f"render: {type(e).__name__}: {e}"
    renderer.close()
    del renderer

    # --- phase 2: score the rendered mp4s (GPU now free for metric models) ---
    import stage2_lean
    for it in items:
        if it.get("error") or "rinfo" not in it:
            it.setdefault("combined", 0.0)
            it.setdefault("metrics", {})
            continue
        try:
            combined, metrics, _ = stage2_lean.score(it["out_mp4"], it["rinfo"], it["segments"], it["eval_idea"])
            it["combined"] = combined
            it["metrics"] = metrics
        except Exception as e:
            it["combined"] = 0.0
            it["metrics"] = {}
            it["error"] = f"score: {type(e).__name__}: {e}"

    with open(args.work, "w") as f:
        json.dump(items, f, indent=2)


if __name__ == "__main__":
    main()
