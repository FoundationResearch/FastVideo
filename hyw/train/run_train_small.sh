#!/usr/bin/env bash
set -euo pipefail

# Small-scale HY-WorldPlay training on the synthetic circle dataset.
# Edit paths below before running.

conda activate alexfv

export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

# --- YOU MUST EDIT THESE ---
MODEL_PATH="/PATH/TO/HY_WORLD_MODEL_DIR"                 # same as used by hyvideo/generate.py --model_path
ACTION_CKPT="/PATH/TO/ACTION.safetensors"                # same as used by hyvideo/generate.py --action_ckpt (a .safetensors file)
TRANSFORMER_DIR="${MODEL_PATH}/transformer/480p_i2v"      # transformer weights dir
AR_ACTION_CKPT="${ACTION_CKPT}"                           # trainer expects a safetensors FILE here (it uses load_file())

TRAIN_JSON="/home/hao_lab/alex/FastVideo/hyw/data/sythcircle_v0_train_for_hyworld.json"
OUT_DIR="/home/hao_lab/alex/FastVideo/outputs/hyworld_sythcircle_small"

# 1 GPU smoke test
NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=0
export MASTER_PORT=29611

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HYWORLD_ROOT="${REPO_ROOT}/hyw/HY-WorldPlay-main"

cd "${HYWORLD_ROOT}"

torchrun \
  --master_port=${MASTER_PORT} \
  --nproc_per_node=${NUM_GPUS} \
  --nnodes 1 \
  trainer/training/ar_hunyuan_w_mem_training_pipeline.py \
  --num_gpus ${NUM_GPUS} \
  --sp_size 1 \
  --tp_size 1 \
  --hsdp_replicate_dim 1 \
  --hsdp_shard_dim ${NUM_GPUS} \
  --cls_name "HunyuanTransformer3DARActionModel" \
  --load_from_dir "${TRANSFORMER_DIR}" \
  --ar_action_load_from_dir "${AR_ACTION_CKPT}" \
  --model_path "${MODEL_PATH}" \
  --pretrained_model_name_or_path "${MODEL_PATH}" \
  --json_path "${TRAIN_JSON}" \
  --causal \
  --action \
  --i2v_rate 0.2 \
  --train_time_shift 3.0 \
  --window_frames 16 \
  --output_dir "${OUT_DIR}" \
  --max_train_steps 50 \
  --train_batch_size 1 \
  --train_sp_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --dataloader_num_workers 0 \
  --learning_rate 1e-5 \
  --mixed_precision "bf16" \
  --checkpointing_steps 25 \
  --weight_decay 1e-4 \
  --max_grad_norm 1.0 \
  --inference_mode False \
  --checkpoints_total_limit 2 \
  --training_cfg_rate 0.0 \
  --not_apply_cfg_solver \
  --dit_precision "fp32" \
  --num_euler_timesteps 50 \
  --ema_start_step 0


