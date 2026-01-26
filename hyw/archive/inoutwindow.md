## in-window / out-window 选 chunk 示例（window_frames=16, latent_T=32）

设：
- `latent_T = 32`（latent index: 0..31）
- 1 个 **chunk = 4 个 latent**
- 所以总共有 **8 个 chunk**
- `window_frames = 16` = **4 个 chunk**

### 1) 原始整段视频（8 chunks）

```text
chunk0: [  0  1  2  3 ]
chunk1: [  4  5  6  7 ]
chunk2: [  8  9 10 11 ]
chunk3: [ 12 13 14 15 ]   <- window 结束位置（16 个 latent）
chunk4: [ 16 17 18 19 ]
chunk5: [ 20 21 22 23 ]
chunk6: [ 24 25 26 27 ]
chunk7: [ 28 29 30 31 ]
```

---

### 2) in-window（连续取前 window_frames=16 个 latent）

in-window 直接取 `latent[0:16]`，也就是前 4 个 chunk：

```text
喂给模型（in-window）:
[  0  1  2  3 ][  4  5  6  7 ][  8  9 10 11 ][ 12 13 14 15 ]
  chunk0          chunk1          chunk2          chunk3
```

---

### 3) out-window（举一个具体采样例子）

out-window 会在 `current_frame_idx ∈ {16, 20, 24, 28}` 里（chunk 对齐，步长 4）随机抽一个“当前 chunk 起点”。

这里举例：假设抽到
- `current_frame_idx = 24`（也就是 **chunk6** 的起点）
- `pred_latent_size = 4`（当前要预测 1 个 chunk）
- `temporal_context_size = 12`（强制包含最近 12 个 latent = 最近 3 个 chunk：chunk3/4/5）
- `memory_frames = 20`（历史 memory latent 数上限；同时永远包含 chunk0 作为 context）

那么“原始时间轴上”的语义是：

```text
原始时间轴:
[  0  1  2  3 ][  4  5  6  7 ][  8  9 10 11 ][ 12 13 14 15 ][ 16 17 18 19 ][ 20 21 22 23 ][ 24 25 26 27 ][ 28 29 30 31 ]
  chunk0          chunk1          chunk2          chunk3          chunk4          chunk5          chunk6          chunk7
                                                                                     ^^^^^^^^^^^^  ^^^^^^^^^^^^
                                                                                     recent ctx     current chunk start
```

最终 **喂给模型的序列不是连续的**，而是“打包/重排”后的短序列，形如：

```text
喂给模型（out-window，打包后的一种可能）:
[  0  1  2  3 ] + [  8  9 10 11 ] + [ 12 13 14 15 ][ 16 17 18 19 ][ 20 21 22 23 ] + [ 24 25 26 27 ]
  chunk0           chunk2            chunk3          chunk4          chunk5           chunk6

解释:
- chunk0: 永远强制加入（first chunk as context）
- chunk3/4/5: temporal_context_size=12 强制加入的“最近上下文”（3 个 chunk）
- chunk2: 举例表示 FOV overlap selector 从更早历史里额外挑到的一个 chunk（真实训练中可能挑 chunk1 或 chunk2，取决于相机 pose 相似度）
- chunk6: 当前要学/要预测的 chunk（并且 loss 通常只监督最后这个 chunk）
```

> 注意：out-window 的“最终输入长度”大致是 **first chunk + 若干历史 memory + 最近 ctx + 当前 chunk**，不会把 `0..current_frame_idx-1` 全部喂进去。
