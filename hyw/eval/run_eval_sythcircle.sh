#!/usr/bin/env bash
set -euo pipefail

# Eval HY-WorldPlay on sythcircle (train/test split).
# Run from anywhere:
#   bash hyw/eval/run_eval_sythcircle.sh
#
# Common overrides:
#   MODEL_PATH=... ACTION_CKPT=... FINETUNED_CKPT=... SPLIT=test bash hyw/eval/run_eval_sythcircle.sh
#   MAX_SAMPLES=32 NUM_INFERENCE_STEPS=20 SEED=123 bash hyw/eval/run_eval_sythcircle.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# --- Split / dataset ---
: "${SPLIT:=test}"  # train|test
: "${DATASET_ROOT:=hyw/data/sythcircle_v1_125f}"  # relative to repo root

# --- Paths (defaults) ---
# These defaults match the typical output printed by hyw/HY-WorldPlay-main/download_models.py
: "${MODEL_PATH:=~/alex/weights/tencent/HunyuanVideo-1.5}"
: "${ACTION_CKPT:=~/alex/weights/tencent/HY-WorldPlay/ar_model/diffusion_pytorch_model.safetensors}"

# Optional: set to checkpoint dir like:
#   hyw/outputs/hyworld_sythcircle_small_original_official/checkpoint-50
: "${FINETUNED_CKPT:=}"

# --- Eval knobs ---
: "${MAX_SAMPLES:=32}"
: "${NUM_INFERENCE_STEPS:=20}"
: "${SEED:=123}"
: "${DEVICE:=cuda}"
: "${OVERWRITE:=1}"  # 1=overwrite metrics.jsonl/summary.json, 0=append

# Expand "~" manually since parameter expansion doesn't expand it.
MODEL_PATH="${MODEL_PATH/#\~/$HOME}"
ACTION_CKPT="${ACTION_CKPT/#\~/$HOME}"
FINETUNED_CKPT="${FINETUNED_CKPT/#\~/$HOME}"

if [[ "${SPLIT}" != "train" && "${SPLIT}" != "test" ]]; then
  echo "ERROR: SPLIT must be 'train' or 'test' (got: ${SPLIT})" >&2
  exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
  echo "ERROR: MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 1
fi
if [ ! -f "${ACTION_CKPT}" ]; then
  echo "ERROR: ACTION_CKPT (safetensors) not found: ${ACTION_CKPT}" >&2
  exit 1
fi
if [[ -n "${FINETUNED_CKPT}" && ! -d "${FINETUNED_CKPT}" ]]; then
  echo "ERROR: FINETUNED_CKPT is set but does not exist: ${FINETUNED_CKPT}" >&2
  exit 1
fi

OUT_DIR="${REPO_ROOT}/hyw/eval/result/${SPLIT}"
if [[ -n "${FINETUNED_CKPT}" ]]; then
  ckpt_name="$(basename "${FINETUNED_CKPT}")"
  out_tag="${ckpt_name}"
  OUT_DIR="${REPO_ROOT}/hyw/eval/result/${SPLIT}_${out_tag}"
fi

cmd=(
  python "${REPO_ROOT}/hyw/eval/eval_sythcircle_split.py"
  --split "${SPLIT}"
  --dataset_root "${DATASET_ROOT}"
  --model_path "${MODEL_PATH}"
  --action_ckpt "${ACTION_CKPT}"
  --out_dir "${OUT_DIR}"
  --max_samples "${MAX_SAMPLES}"
  --num_inference_steps "${NUM_INFERENCE_STEPS}"
  --seed "${SEED}"
  --device "${DEVICE}"
)

if [[ -n "${FINETUNED_CKPT}" ]]; then
  cmd+=( --finetuned_ckpt "${FINETUNED_CKPT}" )
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  cmd+=( --overwrite )
fi

echo "Running:"
printf '  %q' "${cmd[@]}"
echo

"${cmd[@]}"

echo
echo "Done. Results in: ${OUT_DIR}"
echo "Open: ${OUT_DIR}/<sample_id>/side_by_side.mp4"


