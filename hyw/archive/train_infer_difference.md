### 一针见血：两套 forward 的“运算差异”到底在哪里（可直接定位到代码行）

你现在看到的现象（trainer transformer 推理“对了”，eval transformer 推理“不一致”）不是由 shift/CFG 这种参数造成的，而是 **eval 路径根本没有执行训练时那条 forward 的同一段计算图**。

下面按“真正的 forward 运算差异”列出，全部对应到确定的函数实现。

## 1) 训练用的 forward：图像流 + 文本流 + 单流（single_blocks）都会参与每一步计算

训练用的模型类是：
- `ARHunyuanVideo_1_5_DiffusionTransformer`（trainer-side）
- 文件：`hyw/HY-WorldPlay-main/trainer/models/hyvideo/models/transformers/ar_action_hunyuanvideo_1_5_transformer.py`

它的 `forward()` 在每一次调用里做了这些关键计算（都是实打实的算子差异）：
- **显式计算 text stream**（txt_in / byt5 / vision token reorder），并且在进入 block 之前会把 padding token 直接裁掉：
  - `txt = txt[text_mask.bool(...)]`（只保留有效 token）
- **double_blocks + single_blocks 都会跑**：
  - 先 `for block in self.double_blocks: img, txt = block(...)`
  - 然后 `x = cat(img, txt)` 进入 `for block in self.single_blocks: x = block(...)`
  - 最后 `final_layer(img, vec)`

对应实现见：

```750:963:hyw/HY-WorldPlay-main/trainer/models/hyvideo/models/transformers/ar_action_hunyuanvideo_1_5_transformer.py
def forward(...):
    ...
    txt = self.txt_in(...)
    ...
    txt = txt[text_mask.bool().to(txt.device)].unsqueeze(0)
    for index, block in enumerate(self.double_blocks):
        img, txt = block(...)
    x = torch.cat((img, txt), 1)
    for index, block in enumerate(self.single_blocks):
        x = block(...)
    img = x[:, :img_seq_len, ...]
    img = self.final_layer(img, vec)
    ...
```

## 2) eval transformer 在推理时走的 forward：`forward_txt()` 只跑一次；每个 denoise step 只跑 `forward_vision()`（没有 single_blocks）

eval 使用的是 `HunyuanVideo_1_5_Pipeline`（hyvideo-side），它在 AR rollout 里明确做了两步：

### 2.1 先缓存文本 K/V（只做一次，不随 denoise step 更新）

在 `worldplay_video_pipeline.py::ar_rollout()`，它会先调用 transformer 的 `ar_txt_inference=True` 路径来 cache 文本 K/V：

```1038:1070:hyw/HY-WorldPlay-main/hyvideo/pipelines/worldplay_video_pipeline.py
t_expand_txt = torch.tensor([0]).to(device).to(latents.dtype)
self._kv_cache = self.transformer(
    bi_inference=False,
    ar_txt_inference=True,
    ar_vision_inference=False,
    timestep_txt=t_expand_txt,
    text_states=prompt_embeds[...],
    encoder_attention_mask=prompt_mask[...],
    vision_states=vision_states[...],
    ...
    kv_cache=self._kv_cache,
    cache_txt=True,
)
```

这一步对应到 transformer 里的：
- `worldplay_1_5_transformer.py::forward_txt()`（hyvideo-side）
- 它会 mask 文本 token，然后在 **double_blocks** 内产生/缓存 `k_txt/v_txt`：

```1245:1299:hyw/HY-WorldPlay-main/hyvideo/models/transformers/worldplay_1_5_transformer.py
def forward_txt(..., cache_txt=False):
    txt, text_mask, vec_txt = self.get_text_and_mask(...)
    txt = txt[text_mask.bool().to(txt.device)].unsqueeze(0)
    for index, block in enumerate(self.double_blocks):
        txt, t_kv = block(..., ar_txt_inference=True, kv_cache=kv_cache, cache_txt=cache_txt)
        if cache_txt:
            _kv_cache_new[index]["k_txt"] = t_kv["k_txt"]
            _kv_cache_new[index]["v_txt"] = t_kv["v_txt"]
```

### 2.2 每个 denoise step 只跑 `forward_vision()`（只走图像流 + KV-cache + rope slice；不跑 single_blocks）

在 `ar_rollout()` 的每个 timestep，它调用：

```1183:1223:hyw/HY-WorldPlay-main/hyvideo/pipelines/worldplay_video_pipeline.py
noise_pred = self.transformer(
    bi_inference=False,
    ar_txt_inference=False,
    ar_vision_inference=True,
    hidden_states=latents_concat,
    timestep=timestep_input,
    ...
    kv_cache=self._kv_cache,
    cache_vision=False,
    rope_temporal_size=...,
    start_rope_start_idx=...,
)[0]
```

这会进入：
- `worldplay_1_5_transformer.py::forward_vision()`（hyvideo-side）

`forward_vision()` 的关键点（与训练 forward 的计算图不同）：
- **没有 text stream 输入**（不接收 text_states，也不计算 txt token）
- 通过 `kv_cache` 把之前缓存的 `k_txt/v_txt` 注入 block（这和训练的“显式 txt token + single_blocks”不是同一路径）
- 使用 `rope_temporal_size` / `start_rope_start_idx` 对 RoPE 位置编码做 slicing（训练 forward 不存在这套 slicing 逻辑）
- **只跑 double_blocks**，然后直接 `final_layer`，没有 single_blocks：

```1301:1423:hyw/HY-WorldPlay-main/hyvideo/models/transformers/worldplay_1_5_transformer.py
def forward_vision(..., kv_cache=None, rope_temporal_size=4, start_rope_start_idx=0):
    ...
    freqs_cos, freqs_sin = self.get_rotary_pos_embed((rope_temporal_size, th, tw))
    freqs_cos = freqs_cos[start_index:end_index, ...]
    freqs_sin = freqs_sin[start_index:end_index, ...]
    ...
    for index, block in enumerate(self.double_blocks):
        img, vision_kv = block(..., ar_vision_inference=True, kv_cache=kv_cache, cache_vision=cache_vision)
    img = self.final_layer(img, vec)
    return (img, features_list)
```

**这就是“为什么你 eval transformer 不会跟 trainer transformer 一致”的硬原因**：  
训练时每步 forward 的计算图包含 `single_blocks`、显式 txt token 流；而 eval 推理时每步 denoise 的计算图是 `forward_vision()`，它完全不跑 `single_blocks`，并且文本条件是通过 KV-cache 注入而不是显式 txt token 与单流融合。

## 3) 这解释了两个实测现象（完全对上）

- `compare_forward_train_vs_hyvideo.py` 用 “trainer forward vs hyvideo forward_bi” 做同输入对齐时，差异很大：因为两个 forward 的计算图本来就不同（是否裁剪 txt、是否跑 single_blocks、是否带 kv_cache/rope slicing 等）。
- `infer_with_trainer_transformer_50step.py` 用 trainer transformer 跑 50-step 得到“对了”的视频：因为它走的就是训练时那条 forward 语义。


