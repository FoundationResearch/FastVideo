#!/bin/bash

num_gpus=1
export FASTVIDEO_ATTENTION_BACKEND=
export MODEL_BASE=Wan-AI/Wan2.1-T2V-1.3B-Diffusers
# Enable SVDQuant runtime and disable FSDP for testing
export FASTVIDEO_SVDQ_ENABLE=1
# export MODEL_BASE=hunyuanvideo-community/HunyuanVideo
# You can either use --prompt or --prompt-txt, but not both.
fastvideo generate \
    --model-path $MODEL_BASE \
    --sp-size $num_gpus \
    --tp-size 1 \
    --num-gpus $num_gpus \
    --dit-cpu-offload False \
    --use-fsdp-inference False \
    --vae-cpu-offload False \
    --text-encoder-cpu-offload True \
    --pin-cpu-memory False \
    --height 480 \
    --width 832 \
    --num-frames 16 \
    --num-inference-steps 50 \
    --fps 16 \
    --guidance-scale 6.0 \
    --flow-shift 8.0 \
    --prompt-txt assets/prompt.txt \
    --svdq-enable True \
    --svdq-rank 32 \
    --svdq-w-percentile 0.999 \
    --svdq-act-unsigned False \
    --negative-prompt "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards" \
    --seed 1024 \
    --output-path outputs_video_svdquant/