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

# 推荐：带上 HF token（并且你需要已获 gated 模型 black-forest-labs/FLUX.1-Redux-dev 的访问权限）
python download_models.py --weights_root ~/alex/weights --hf_token "$HF_TOKEN"
```

脚本结束会打印：
- `MODEL_PATH=...`  （这是 HunyuanVideo-1.5 的本地目录）
- `AR_ACTION_MODEL_PATH=.../diffusion_pytorch_model.safetensors`（这是 action `.safetensors` 文件）

把它们记下来，后面会用到：
- `--model_path` / `MODEL_PATH` ← 用打印出来的 `MODEL_PATH`
- `--action_ckpt` / `ACTION_CKPT` ← 用打印出来的 `AR_ACTION_MODEL_PATH`

### 重要检查：SigLIP vision encoder（必须有）

`precompute_latents.py` / 推理 / eval 会在创建 pipeline 时检查：

- `${MODEL_PATH}/vision_encoder/siglip`

如果缺了（你现在就是缺这个），先解决它：重新跑 Step 1（带 token + 已获访问权限），直到目录存在为止。

## Step 2：预计算 `latent.pt`（目的：把视频编码成训练期要读的 `.pt`）

```bash
cd /home/hao_lab/alex/FastVideo
python hyw/train/precompute_latents.py \
  --raw_manifest /home/hao_lab/alex/FastVideo/hyw/data/sythcircle_v0/manifest_raw_train.json \
  --model_path /mnt/fast-disks/hao_lab/alex/weights/tencent/HunyuanVideo-1.5 \
  --action_ckpt /mnt/fast-disks/hao_lab/alex/weights/tencent/HY-WorldPlay/ar_model/diffusion_pytorch_model.safetensors \
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

1) （可选）编辑 `hyw/train/run_train_small.sh` 里的默认路径，或用环境变量覆盖：

```bash
MODEL_PATH=/mnt/fast-disks/hao_lab/alex/weights/tencent/HunyuanVideo-1.5 \
ACTION_CKPT=/mnt/fast-disks/hao_lab/alex/weights/tencent/HY-WorldPlay/ar_model/diffusion_pytorch_model.safetensors \
bash hyw/train/run_train_small.sh
```

2) 然后运行：

```bash
bash hyw/train/run_train_small.sh
```

备注（第一跑建议）：
- 用 `--training_cfg_rate 0.0`（否则需要额外 negative prompt 文件）
- `--max_train_steps 50` 先 smoke-test；跑通后再加大


