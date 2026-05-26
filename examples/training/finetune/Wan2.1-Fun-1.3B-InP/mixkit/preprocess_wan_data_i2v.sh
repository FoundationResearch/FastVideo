#!/bin/bash
# Preprocess the full Mixkit-Src dataset (FastVideo/Mixkit-Src) into i2v parquet
# (video VAE latent + first-frame latent + CLIP feature + text embedding).
# Run from the FastVideo/ repo root.

GPU_NUM=1 # v1_preprocess.py asserts num_gpus==1 (single-GPU only)
MODEL_PATH="weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers"
MODEL_TYPE="wan"
# merge_local.txt points at: data/Mixkit-Src,data/Mixkit-Src/video2caption_replace.json
DATA_MERGE_PATH="data/Mixkit-Src/merge_local.txt"
OUTPUT_DIR="data/mixkit_processed_i2v_1_3b_inp/"

torchrun --nproc_per_node=$GPU_NUM \
    fastvideo/pipelines/preprocess/v1_preprocess.py \
    --model_path $MODEL_PATH \
    --data_merge_path $DATA_MERGE_PATH \
    --preprocess_video_batch_size 8 \
    --seed 42 \
    --max_height 480 \
    --max_width 832 \
    --num_frames 77 \
    --dataloader_num_workers 0 \
    --output_dir=$OUTPUT_DIR \
    --train_fps 16 \
    --samples_per_file 8 \
    --flush_frequency 8 \
    --video_length_tolerance_range 5 \
    --preprocess_task "i2v"
