#!/bin/bash
# Evaluate the finetuned checkpoint-2000. Instead of resuming the full training
# state (whose optimizer state fails to load), we point model_path at a merged
# model dir = base components + the finetuned transformer weights, and run ONE
# step. The pipeline runs an initial validation at step 0 BEFORE training, on
# the loaded (finetuned) weights, generating diverse mixkit validation videos
# to wandb. Run from the FastVideo/ repo root.

source ../nanoVideo/env.sh
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export WANDB_NAME="eval-ckpt2000-mixkit-val"
export TOKENIZERS_PARALLELISM=false

# Merged model dir: base vae/text_encoder/image_encoder/tokenizer + finetuned transformer
MODEL_PATH="data/mixkit_processed_i2v_1_3b_inp/outputs/finetuned_model_ckpt2000"
DATA_DIR="data/mixkit_processed_i2v_1_3b_inp/combined_parquet_dataset/"
VALIDATION_DATASET_FILE="examples/training/finetune/Wan2.1-Fun-1.3B-InP/mixkit/validation.json"
NUM_GPUS=4

training_args=(
  --tracker_project_name "fastvideo-i2v-distill"
  --output_dir "data/mixkit_processed_i2v_1_3b_inp/outputs/wan_i2v_eval_ckpt2000"
  --max_train_steps 1
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 8
  --num_height 480
  --num_width 832
  --num_frames 77
  --enable_gradient_checkpointing_type "full"
)

# Must match the checkpoint's parallel layout
parallel_args=(
  --num_gpus $NUM_GPUS
  --sp_size 4
  --tp_size 1
  --hsdp_replicate_dim 1
  --hsdp_shard_dim 4
)

model_args=(
  --model_path $MODEL_PATH
  --pretrained_model_name_or_path $MODEL_PATH
)

dataset_args=(
  --data_path "$DATA_DIR"
  --dataloader_num_workers 1
)

# Validation runs once at the resumed step (2000) on the loaded weights.
validation_args=(
  --log_validation
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 100
  --validation_sampling_steps "40"
  --validation_guidance_scale "6.0"
)

optimizer_args=(
  --learning_rate 2e-5
  --mixed_precision "bf16"
  --weight_only_checkpointing_steps 100000
  --training_state_checkpointing_steps 100000
  --weight_decay 1e-4
  --max_grad_norm 1.0
)

miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 3
  --training_cfg_rate 0.1
  --multi_phased_distill_schedule "4000-1"
  --not_apply_cfg_solver
  --dit_precision "fp32"
  --num_euler_timesteps 50
  --ema_start_step 0
)

torchrun \
  --nnodes 1 \
  --nproc_per_node $NUM_GPUS \
    fastvideo/training/wan_i2v_training_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}"
