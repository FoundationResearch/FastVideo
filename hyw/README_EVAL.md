## Can I use the `test/` split for evaluation?

Yes — **you can use `hyw/data/sythcircle_v0/test` as evaluation** *as long as you do not tune / iterate hyperparams on it*.

What we do here:
- **Input**: use **test `action.json` + `pose.json`** (control signals) + a **reference image** (GT frame 0) to run HY-WorldPlay inference.
- **Output**: save predicted videos so you can eyeball them.
- **Metric**: compute simple **pixel-space** differences vs the GT test video (L1 / MSE / PSNR).

## Important note about frame count (64 → 61)

HY-WorldPlay inference expects frame count aligned to latent steps:

\[
F = 4\cdot(L-1)+1
\]

Your test videos are 64 frames, but the closest valid length for \(L=16\) is **61 frames**.

So the eval script will automatically:
- **truncate GT to 61 frames**
- generate **61-frame** predictions
- compute metrics on those 61 frames

## What this eval produces

For each sample:
- `gt.mp4`: ground-truth video (possibly truncated to 61 frames)
- `pred.mp4`: model output video using **test actions/poses**
- `side_by_side.mp4`: left=GT, right=pred (for quick visual inspection)

And global files:
- `metrics.jsonl`: one JSON per sample with metrics + output paths
- `summary.json`: averages over evaluated samples

## Run evaluation

### Step 0: Activate env

```bash
conda activate alexfv
```

### Step 1: Make sure you have base models + action ckpt

Run the download once (it prints the paths you need).
**Do NOT skip the vision encoder** for eval, because `create_pipeline()` requires
`MODEL_PATH/vision_encoder/siglip` to exist.

```bash
cd /home/hao_lab/alex/FastVideo/hyw/HY-WorldPlay-main
python download_models.py --hf_token <your_token>
```

Record:
- `MODEL_PATH=...`
- `AR_ACTION_MODEL_PATH=.../diffusion_pytorch_model.safetensors`

### Step 2: (Optional) pick a fine-tuned checkpoint

If you trained and have an output dir like:
- `.../checkpoint-50/transformer/diffusion_pytorch_model.safetensors`

Then you can pass `--finetuned_ckpt .../checkpoint-50`.

If you don’t pass it, eval runs **base** weights + `ACTION_CKPT`.

### Step 3: Run eval on test split

```bash
cd /home/hao_lab/alex/FastVideo

python hyw/eval/eval_sythcircle_test.py \
  --raw_manifest /home/hao_lab/alex/FastVideo/hyw/data/sythcircle_v0/manifest_raw_test.json \
  --model_path <MODEL_PATH_FROM_DOWNLOAD> \
  --action_ckpt <AR_ACTION_MODEL_PATH_FROM_DOWNLOAD> \
  --finetuned_ckpt <OPTIONAL_OUTPUT_DIR/checkpoint-50> \
  --out_dir /home/hao_lab/alex/FastVideo/outputs/eval_sythcircle_test \
  --max_samples 4 \
  --num_inference_steps 20
```

Then open:
- `outputs/eval_sythcircle_test/<sample_id>/side_by_side.mp4`


