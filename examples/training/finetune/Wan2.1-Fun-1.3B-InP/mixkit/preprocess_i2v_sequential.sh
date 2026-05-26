#!/bin/bash
# Sequential (1-at-a-time) completion of i2v preprocessing. The preprocessing
# has a memory leak (~2.8GB/batch); a single 1371-sample shard peaks near the
# 512GB Slurm budget, so shards MUST run one at a time. Redo any shard among
# 1,2,3 that did not finish (log lacks the 172/172 completion marker), then
# merge shard0..3 into the final combined_parquet_dataset.
# Run from the FastVideo/ repo root, AFTER the resume orchestration has exited.
set -u

MODEL_PATH="weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers"
SHARD_DIR="data/_mixkit_shards"
FINAL_DIR="data/mixkit_processed_i2v_1_3b_inp/combined_parquet_dataset"

is_complete() {  # complete if the shard log reached 172/172 (or N/N) and has parquet
  local i=$1
  local w="$SHARD_DIR/shard${i}/combined_parquet_dataset/worker_0"
  [ -d "$w" ] && [ "$(ls "$w"/*.parquet 2>/dev/null | wc -l)" -gt 0 ] || return 1
  grep -qE "Processing videos: 100%" "$SHARD_DIR/shard${i}.log" 2>/dev/null
}

run_shard() {
  local i=$1
  echo "[seq] (re)running shard$i on GPU 0 ..."
  rm -rf "$SHARD_DIR/shard${i}/combined_parquet_dataset"
  CUDA_VISIBLE_DEVICES=0 torchrun \
      --nnodes 1 --nproc_per_node 1 --master_port 29510 \
      fastvideo/pipelines/preprocess/v1_preprocess.py \
      --model_path "$MODEL_PATH" \
      --data_merge_path "$SHARD_DIR/merge${i}.txt" \
      --preprocess_video_batch_size 8 \
      --seed 42 \
      --max_height 480 --max_width 832 --num_frames 77 \
      --dataloader_num_workers 0 \
      --output_dir "$SHARD_DIR/shard${i}/" \
      --train_fps 16 --samples_per_file 8 --flush_frequency 8 \
      --video_length_tolerance_range 5 \
      --preprocess_task "i2v" \
      > "$SHARD_DIR/shard${i}.log" 2>&1
  echo "[seq] shard$i exit=$?"
}

for i in 1 2 3; do
  if is_complete "$i"; then
    echo "[seq] shard$i already complete, skipping."
  else
    run_shard "$i"
    is_complete "$i" || { echo "[seq] ERROR: shard$i still incomplete after rerun"; exit 1; }
  fi
done

echo "[seq] all shards complete; merging shard0..3 -> $FINAL_DIR"
rm -rf "$FINAL_DIR"; mkdir -p "$FINAL_DIR"
for i in 0 1 2 3; do
  cp -r "$SHARD_DIR/shard${i}/combined_parquet_dataset/worker_0" "$FINAL_DIR/worker_${i}"
  echo "  shard$i -> worker_${i} ($(ls "$FINAL_DIR/worker_${i}"/*.parquet | wc -l) parquet)"
done
rows=$(python - "$FINAL_DIR" <<'PY'
import sys, glob, pyarrow.parquet as pq
files = glob.glob(sys.argv[1]+"/**/*.parquet", recursive=True)
print(sum(pq.read_metadata(f).num_rows for f in files))
PY
)
echo "[seq] DONE. total samples = $rows  (target ~5486)"
