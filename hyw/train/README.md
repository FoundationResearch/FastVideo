# HY-WorldPlay 训练（用 hyw/data 的合成数据）

目标：把 `hyw/data/sythcircle_v0/` 里已生成好的 mp4+pose+action，变成 HY-WorldPlay 训练能直接读取的 `latent.pt` + `json_path`，然后启动一个小规模训练跑通链路。

## Step 0：激活环境（目的：确保依赖齐全）

```bash
conda activate alexfv
```

## Step 1：预计算 `latent.pt`（目的：把视频编码成训练期要读的 `.pt`）

```bash
python hyw/train/precompute_latents.py \
  --raw_manifest /home/hao_lab/alex/FastVideo/hyw/data/sythcircle_v0/manifest_raw_train.json \
  --model_path /PATH/TO/HY_WORLD_MODEL_DIR \
  --action_ckpt /PATH/TO/ACTION.safetensors \
  --transformer_version 480p_i2v \
  --out_root /home/hao_lab/alex/FastVideo/hyw/data/sythcircle_v0_latent_pt/train \
  --max_samples 32
```

输出：`--out_root/sample_00000/latent.pt`（每个样本一个）

## Step 2：生成训练用 `json_path`（目的：把 latent/pose/action 路径拼成训练 manifest）

```bash
python hyw/train/make_training_json.py \
  --raw_manifest /home/hao_lab/alex/FastVideo/hyw/data/sythcircle_v0/manifest_raw_train.json \
  --latent_root /home/hao_lab/alex/FastVideo/hyw/data/sythcircle_v0_latent_pt/train \
  --out_json /home/hao_lab/alex/FastVideo/hyw/data/sythcircle_v0_train_for_hyworld.json \
  --max_samples 32
```

输出：`.../sythcircle_v0_train_for_hyworld.json`

## Step 3：启动训练（目的：跑通 action+camera+memory 的训练 pipeline）

1) 先编辑 `hyw/train/run_train_small.sh` 里的：
- `MODEL_PATH`
- `ACTION_CKPT`

2) 然后运行：

```bash
bash hyw/train/run_train_small.sh
```

备注（第一跑建议）：
- 用 `--training_cfg_rate 0.0`（否则需要额外 negative prompt 文件）
- `--max_train_steps 50` 先 smoke-test；跑通后再加大


