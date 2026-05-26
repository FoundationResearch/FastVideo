#!/bin/bash
# Resume i2v preprocessing for the shards that were OOM-killed (1,2,3) under a
# 512GB Slurm memory budget. shard0 already completed. Run at most 2 shards
# concurrently (~2x200GB peak < 512GB) to avoid the OOM that killed the 4-way run.
# After all shards exist, merge into one combined_parquet_dataset.
# Run from the FastVideo/ repo root.
set -u

MODEL_PATH="weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers"
SHARD_DIR="data/_mixkit_shards"
FINAL_DIR="data/mixkit_processed_i2v_1_3b_inp/combined_parquet_dataset"
REDO="1 2 3"
MAXJOBS=2

run_shard() {
  local i=$1
  rm -rf "$SHARD_DIR/shard${i}/combined_parquet_dataset"   # drop partial output
  CUDA_VISIBLE_DEVICES=$i torchrun \
      --nnodes 1 --nproc_per_node 1 --master_port $((29501+i)) \
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
  echo "[resume] shard$i exit=$?"
}

echo "[resume] re-running shards: $REDO (max $MAXJOBS concurrent) ..."
for i in $REDO; do
  run_shard "$i" &
  echo "  launched shard$i (GPU $i, pid $!)"
  while [ "$(jobs -r | wc -l)" -ge "$MAXJOBS" ]; do wait -n; done
done
wait
echo "[resume] all redo shards finished."

# Verify every shard (0..3) has output, then merge.
echo "[resume] merging shard0..3 into $FINAL_DIR ..."
rm -rf "$FINAL_DIR"; mkdir -p "$FINAL_DIR"
fail=0
for i in 0 1 2 3; do
  w="$SHARD_DIR/shard${i}/combined_parquet_dataset/worker_0"
  if [ -d "$w" ] && [ "$(ls "$w"/*.parquet 2>/dev/null | wc -l)" -gt 0 ]; then
    cp -r "$w" "$FINAL_DIR/worker_${i}"
    echo "  shard$i -> worker_${i} ($(ls "$FINAL_DIR/worker_${i}"/*.parquet | wc -l) parquet)"
  else
    echo "  ERROR: shard$i missing output"; fail=1
  fi
done
[ "$fail" -ne 0 ] && { echo "[resume] merge incomplete, leaving shards in place."; exit 1; }

rows=$(python - "$FINAL_DIR" <<'PY'
import sys, glob, pyarrow.parquet as pq
files = glob.glob(sys.argv[1]+"/**/*.parquet", recursive=True)
print(sum(pq.read_metadata(f).num_rows for f in files))
PY
)
echo "[resume] DONE. total samples = $rows  (target ~5486)"
