#!/bin/bash
# 4-way sharded i2v preprocessing of Mixkit-Src across 4 GPUs.
# v1_preprocess.py is single-GPU only (asserts WORLD_SIZE==1), so we run 4
# independent single-GPU processes (one per GPU) over disjoint caption shards,
# then merge their parquet outputs into one combined_parquet_dataset.
# Run from the FastVideo/ repo root.
set -u

MODEL_PATH="weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers"
SRC_DIR="data/Mixkit-Src"
SRC_JSON="$SRC_DIR/video2caption_replace.json"
SHARD_DIR="data/_mixkit_shards"
FINAL_DIR="data/mixkit_processed_i2v_1_3b_inp/combined_parquet_dataset"
NSHARD=4

rm -rf "$SHARD_DIR" "data/mixkit_processed_i2v_1_3b_inp"
mkdir -p "$SHARD_DIR"

echo "[orch] splitting $SRC_JSON into $NSHARD shards ..."
python - "$SRC_JSON" "$SHARD_DIR" "$SRC_DIR" "$NSHARD" <<'PY'
import json, sys, os
src_json, shard_dir, src_dir, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
data = json.load(open(src_json))
for i in range(n):
    shard = data[i::n]  # round-robin split (balanced across categories)
    sj = os.path.join(shard_dir, f"shard{i}.json")
    json.dump(shard, open(sj, "w"))
    with open(os.path.join(shard_dir, f"merge{i}.txt"), "w") as f:
        f.write(f"{src_dir},{sj}\n")
    print(f"  shard{i}: {len(shard)} videos -> {sj}")
PY

echo "[orch] launching $NSHARD single-GPU preprocessing processes ..."
pids=()
for i in $(seq 0 $((NSHARD-1))); do
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
      > "$SHARD_DIR/shard${i}.log" 2>&1 &
  pids+=($!)
  echo "  shard$i -> GPU $i, pid ${pids[$i]}, log $SHARD_DIR/shard${i}.log"
done

echo "[orch] waiting for all shards ..."
fail=0
for i in $(seq 0 $((NSHARD-1))); do
  if wait ${pids[$i]}; then
    echo "  shard$i DONE"
  else
    echo "  shard$i FAILED (see $SHARD_DIR/shard${i}.log)"; fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  echo "[orch] one or more shards failed; NOT merging."; exit 1
fi

echo "[orch] merging shard outputs into $FINAL_DIR ..."
mkdir -p "$FINAL_DIR"
for i in $(seq 0 $((NSHARD-1))); do
  w="$SHARD_DIR/shard${i}/combined_parquet_dataset/worker_0"
  if [ -d "$w" ]; then
    mv "$w" "$FINAL_DIR/worker_${i}"
    echo "  merged shard$i -> $FINAL_DIR/worker_${i} ($(ls "$FINAL_DIR/worker_${i}"/*.parquet 2>/dev/null | wc -l) parquet)"
  else
    echo "  WARN: shard$i has no worker_0 output"
  fi
done

total=$(find "$FINAL_DIR" -name '*.parquet' | wc -l)
rows=$(python - "$FINAL_DIR" <<'PY'
import sys, glob, pyarrow.parquet as pq
files = glob.glob(sys.argv[1]+"/**/*.parquet", recursive=True)
print(sum(pq.read_metadata(f).num_rows for f in files))
PY
)
echo "[orch] DONE. $total parquet files, $rows total samples in $FINAL_DIR"
rm -rf "$SHARD_DIR"
