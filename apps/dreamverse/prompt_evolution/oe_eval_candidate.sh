#!/usr/bin/env bash
# Evaluate ONE candidate dreamverse rewriter system prompt on ONE user idea:
#   (1) rewrite the idea -> 6 segment prompts (Groq gpt-oss-120b, candidate system prompt)
#   (2) render a 30s rollout via the LTX2 WS server (curated, enhancement OFF) -> mp4 + seam frames
#   (3) score with stage2_video_eval.py (vbench + boundary.video/audio + audio.noise)
# Prints OE_COMBINED_SCORE=<float> and a {combined_score, metrics, artifacts} JSON blob.
#
# Usage: bash oe_eval_candidate.sh <candidate_system_prompt.md> "<user_idea>" <out_dir>
# Run on the GPU node (server is local) e.g. via: srun --jobid=4756 --overlap bash oe_eval_candidate.sh ...
set -euo pipefail

CAND="$1"; IDEA="$2"; OUT="$3"
DV_PY=/home/hal-shao/FastVideo-dreamverse/.venv/bin/python
CLONE=/home/hal-shao/FastVideo-openevolve
PE="$CLONE/apps/dreamverse/prompt_evolution"
ENVFILE=/home/hal-shao/FastVideo-dreamverse/.env

mkdir -p "$OUT"
set -a; source "$ENVFILE"; set +a              # GROQ_API_KEY, ffmpeg codec settings, etc.
export PYTHONPATH="$CLONE"                       # import fastvideo == the clone (has boundary metrics)
export LD_LIBRARY_PATH="/home/shared-bin/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}"  # SQUIM / torchcodec

echo "=== [1/3] rewrite (Groq gpt-oss-120b, candidate system prompt) ==="
"$DV_PY" "$PE/oe_rewrite.py" "$CAND" "$IDEA" "$OUT"

echo "=== [2/3] render (LTX2 server, curated, enhancement OFF) ==="
"$DV_PY" "$PE/gen_curated_rollout.py" "$OUT/segment_prompts.json" "$OUT"

echo "=== [3/3] score (vbench + boundary.video/audio + audio.noise) ==="
MP4="$OUT/rollout_final.mp4"
SEAMS=$("$DV_PY" -c "import json; print(','.join(str(x) for x in json.load(open('$OUT/manifest.json'))['seam_frames']))")
echo "[eval] seams=$SEAMS"
CUDA_VISIBLE_DEVICES="${OE_EVAL_GPU:-1}" "$DV_PY" "$PE/stage2_video_eval.py" \
    "$MP4" --seams "$SEAMS" --prompt "$IDEA" --out "$OUT/eval.json"

echo "=== RESULT ==="
"$DV_PY" - "$OUT/eval.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print("OE_COMBINED_SCORE=" + str(r["combined_score"]))
print(json.dumps({"combined_score": r["combined_score"], "metrics": r["metrics"],
                  "artifacts": {k: v for k, v in r["artifacts"].items() if k != "metrics_guide"}}, indent=2))
PY
