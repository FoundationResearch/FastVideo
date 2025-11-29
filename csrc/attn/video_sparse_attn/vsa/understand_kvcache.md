### Causal + VSA 推理：KV cache 设计（简版）

我们只关心 **推理 / self‑forcing 生成**，训练继续用现有非 causal VSA 即可。

---

### 1. 为什么可以直接 append？

设 patch 后 3D 形状为 `(T', H', W')`，时间维按 4‑frame block 递增（无 padding、无打乱），  
每个 block 在时间上对应一段连续的 `T'_{\text{block}}`。

- `get_tile_partition_indices` 按 `(t_tile, h_tile, w_tile)` 扫描网格；
- 如果对「第 1 个 block」「第 2 个 block」分别单独做 tile，得到的一维序列是 `tile_0`、`tile_1`；
- 再在一维上做 `tile_0 ++ tile_1`，等价于对 8 帧整体做一次 tile。

同理，`construct_variable_block_sizes` 只是在每个 tile 上数「有效 token 个数」：

- 在 `(H', W')` 和 `VSA_TILE_SIZE` 不变的前提下，
- 「每个 block 自己算 `variable_block_sizes_block` 再拼起来」  
  = 「对整个 prefix 一次性算 `variable_block_sizes_prefix`」。

**结论**：  
对每个 4‑frame block：

- tiled K/V 可以 append；
- 对应的 `variable_block_sizes_block` 也可以 append；
- 得到的结果与对整个 prefix 一次性 tile + 一次性算 `variable_block_sizes` 完全一致。

---

### 2. KV cache 里具体存什么？

我们只存 **已经 tile+pad 之后的一维 K/V**，以及对应的 tile 信息：

```python
kv_cache = {
    "k": k_tiled_cache,              # [B, L_tiled_total, num_heads, head_dim]
    "v": v_tiled_cache,              # [B, L_tiled_total, num_heads, head_dim]
    "variable_block_sizes": vbs_all, # [num_tiles_total]
    "num_cached_blocks": int,        # 已生成的 4‑frame block 数
    "dit_seq_shape_block": (T_block_p, H_prime, W_prime),  # 单 block patch 形状
}
```

对新来的一个 block：

1. 用 `dit_seq_shape_block` 计算：
   - `num_tiles_block`、`tile_partition_indices_block`、`variable_block_sizes_block`、`non_pad_index_block`；
2. 对 `k_block_rope / v_block` 做 tile，得到 `k_tiled_block / v_tiled_block`；
3. 在 cache 里简单 append：

```python
if kv_cache is None:
    k_tiled_cache = k_tiled_block
    v_tiled_cache = v_tiled_block
    vbs_all = variable_block_sizes_block
    num_cached_blocks = 1
else:
    k_tiled_cache = torch.cat([kv_cache["k"], k_tiled_block], dim=1)
    v_tiled_cache = torch.cat([kv_cache["v"], v_tiled_block], dim=1)
    vbs_all = torch.cat([kv_cache["variable_block_sizes"],
                         variable_block_sizes_block], dim=0)
    num_cached_blocks = kv_cache["num_cached_blocks"] + 1
```

---

### 3. 每一步推理怎么用 KV cache？

对当前 block：

1. **Q / gate 只算本 block**  
   从 `hidden_states_block` 得到 `q_block, k_block, v_block, gate_block`，  
   对 Q/K 做 qk‑norm + RoPE，得到 `q_block_rope, k_block_rope`。

2. **K/V：tile + append**  
   按第 2 节的流程更新 `kv_cache["k"] / kv_cache["v"] / variable_block_sizes`。

3. **Q / gate：只对当前 block 做 tile**  

```python
q_tiled_block = attn_impl.tile(
    q_block_rope, num_tiles_block, tile_partition_indices_block, non_pad_index_block
)
gate_tiled_block = attn_impl.tile(
    gate_block, num_tiles_block, tile_partition_indices_block, non_pad_index_block
)
```

4. **调用 VSA kernel（利用 `len(q) != len(k)`）**  
   这里 `query` 只覆盖当前 block，`key/value` 覆盖 prefix 全部：

```python
hidden_tiled_out = video_sparse_attn(
    query=q_tiled_block,                  # [B, L_q_block, H, D]
    key=kv_cache["k"],                    # [B, L_k_total, H, D]
    value=kv_cache["v"],                  # 同上
    variable_block_sizes=kv_cache["variable_block_sizes"],
    topk=cur_topk,
    block_size=VSA_TILE_SIZE,
    compress_attn_weight=gate_tiled_block,  # 只对应当前 block 的 Q
)
```

5. **untile 回当前 block 的时间顺序**  
   用当前 block 自己的 `reverse_tile_partition_indices_block / non_pad_index_block`：

```python
hidden_block = attn_impl.untile(
    hidden_tiled_out, reverse_tile_partition_indices_block, non_pad_index_block
)  # [B, L_block, H, D]
```

再 flatten 成 `[B, L_block, dim]`，走 `to_out + residual + FFN` 即可。

---

### 4. 总结给实现看的点

- **不重算 prefix 的 Q**：每步只对当前 4‑frame block 的 Q / gate 做 tile；
- **KV cache 始终在 tile 轴上 append**：`k / v / variable_block_sizes` 都是按 block 拼接；
- **causal 由生成顺序保证**：第 `b` 步调用 kernel 时，K/V 只包含前 `b` 个 block 的 tiled 结果；
- 训练阶段不动，只在 `_forward_inference` / self‑forcing 的 causal 路径里用这套 KV cache 设计。 


