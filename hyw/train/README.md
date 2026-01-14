# HY-WorldPlay 训练（用 hyw/data 的合成数据）

目标：把 `hyw/data/sythcircle_v0/` 里已生成好的 mp4+pose+action，变成 HY-WorldPlay 训练能直接读取的 `latent.pt` + `json_path`，然后启动一个小规模训练跑通链路。

## Step 0：激活环境（目的：确保依赖齐全）

```bash
conda activate alexfv
```

## Step 1：下载模型（目的：拿到 `MODEL_PATH` 和 `ACTION_CKPT`）

在仓库根目录执行：

```bash
conda activate alexfv
cd /home/hao_lab/alex/FastVideo/hyw/HY-WorldPlay-main

# 重要：**不要默认 skip vision encoder**
# 因为 HY-WorldPlay 的 `create_pipeline()` 会强制检查
# `MODEL_PATH/vision_encoder/siglip` 是否存在；缺了会导致
# `precompute_latents.py` / 推理 / eval 在“创建 pipeline”阶段直接报错。
#
# 推荐（有 HF token 且已获 FLUX.1-Redux-dev 访问权限）：
python download_models.py --hf_token <your_token>
#
# 如果你暂时没有权限，可以先下载其它权重：
# python download_models.py --skip_vision_encoder
# 但注意：这种情况下你**无法**运行 `precompute_latents.py` / 推理 / eval（会找不到 siglip）。
```

脚本结束会打印：
- `MODEL_PATH=...`  （这是 HunyuanVideo-1.5 的本地目录）
- `AR_ACTION_MODEL_PATH=.../diffusion_pytorch_model.safetensors`（这是 action `.safetensors` 文件）

把它们记下来，后面会用到：
- `--model_path` / `MODEL_PATH` ← 用打印出来的 `MODEL_PATH`
- `--action_ckpt` / `ACTION_CKPT` ← 用打印出来的 `AR_ACTION_MODEL_PATH`

## Step 2：预计算 `latent.pt`（目的：把视频编码成训练期要读的 `.pt`）

```bash
cd /home/hao_lab/alex/FastVideo
python hyw/train/precompute_latents.py \
  --raw_manifest /home/hao_lab/alex/FastVideo/hyw/data/sythcircle_v0/manifest_raw_train.json \
  --model_path <MODEL_PATH_FROM_STEP1> \
  --action_ckpt <AR_ACTION_MODEL_PATH_FROM_STEP1> \
  --transformer_version 480p_i2v \
  --out_root /home/hao_lab/alex/FastVideo/hyw/data/sythcircle_v0_modelinput/latent_pt/train \
  --max_samples 32
```

输出：`--out_root/sample_00000/latent.pt`（每个样本一个）

## Step 3：生成训练用 `json_path`（目的：把 latent/pose/action 路径拼成训练 manifest）

```bash
cd /home/hao_lab/alex/FastVideo
python hyw/train/make_training_json.py \
  --raw_manifest /home/hao_lab/alex/FastVideo/hyw/data/sythcircle_v0/manifest_raw_train.json \
  --latent_root /home/hao_lab/alex/FastVideo/hyw/data/sythcircle_v0_modelinput/latent_pt/train \
  --out_json /home/hao_lab/alex/FastVideo/hyw/data/sythcircle_v0_modelinput/sythcircle_v0_train_for_hyworld.json \
  --max_samples 32
```

输出：`.../sythcircle_v0_train_for_hyworld.json`

## Step 4：启动训练（目的：跑通 action+camera+memory 的训练 pipeline）

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


