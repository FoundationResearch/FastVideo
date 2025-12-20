import os
import time
import math
from pathlib import Path
from hashlib import md5
from typing import Tuple

import torch
block_sparse_attn=None
import torch
major, minor = torch.cuda.get_device_capability(0)
if major == 9 and minor == 0:# check if H100
    from vsa_cuda import block_sparse_fwd, block_sparse_bwd
    from vsa.block_sparse_wrapper import block_sparse_attn_SM90
    block_sparse_attn = block_sparse_attn_SM90
else:
    from vsa.block_sparse_wrapper import block_sparse_attn_triton
    block_sparse_fwd = None
    block_sparse_bwd = None
    block_sparse_attn = block_sparse_attn_triton

BLOCK_M = 64
BLOCK_N = 64


def torch_attention(q, k, v) -> Tuple[torch.Tensor, torch.Tensor]:
    QK = torch.matmul(q, k.transpose(-2, -1))
    QK /= (q.size(-1)**0.5)

    # Causal mask removed since causal is always false

    QK = torch.nn.functional.softmax(QK, dim=-1)
    output = torch.matmul(QK, v)
    return output, QK


def video_sparse_attn(
    q,
    k,
    v,
    variable_block_sizes,
    topk,
    block_size,
    compress_attn_weight=None,
    q_variable_block_sizes=None,
):
    """
    q: [batch_size, num_heads, seq_len_q, head_dim]
    k: [batch_size, num_heads, seq_len_kv, head_dim]
    v: [batch_size, num_heads, seq_len_kv, head_dim]
    topk: int
    block_size: int or tuple of 3 ints
    video_shape: tuple of (T, H, W)
    compress_attn_weight: [batch_size, num_heads, seq_len, head_dim]
    select_attn_weight: [batch_size, num_heads, seq_len, head_dim]
    q_variable_block_sizes: Optional[Tensor] of shape [num_q_tiles], giving the number of
        valid (non-padded) tokens in each Q tile. If None, we assume Q uses the same
        variable_block_sizes as KV (backwards compatible) **only when** q and kv have the
        same number of tiles. If q and kv have different numbers of tiles, you must pass
        q_variable_block_sizes explicitly.

    NOTE: We assume q, k, v is zero padded!!
    V1 of sparse attention. Include compress attn and sparse attn branch, use average pooling to compress. 
    Assume q, k, v is flattened in this way: [batch_size, num_heads, T//block_size[0], H//block_size[1], W//block_size[2], block_size[0], block_size[1], block_size[2]]
    """

    if isinstance(block_size, int):
        block_size = (block_size, block_size, block_size)

    block_elements = block_size[0] * block_size[1] * block_size[2]
    assert block_elements == 64
    assert q.shape[2] % block_elements == 0
    assert k.shape[2] % block_elements == 0
    assert v.shape[2] % block_elements == 0
    assert k.shape == v.shape, "k and v must have the same shape"

    batch_size, num_heads, seq_len_q, head_dim = q.shape
    seq_len_kv = k.shape[2]
    # variable_block_sizes semantics: KV-side per-tile (per 64-token block) valid token counts.
    # Length must match the number of KV tiles.
    assert variable_block_sizes.numel() == (seq_len_kv // block_elements), (
        f"variable_block_sizes must have length seq_len_kv//{block_elements}, "
        f"but got {variable_block_sizes.numel()} vs {seq_len_kv // block_elements}"
    )
    if compress_attn_weight is not None:
        assert compress_attn_weight.shape == q.shape, (
            f"compress_attn_weight must match q shape {tuple(q.shape)}, "
            f"but got {tuple(compress_attn_weight.shape)}"
        )

    # compress attn
    # Q and KV can have different tile counts. For correctness, Q needs its own
    # per-tile valid token counts if it is also padded.
    num_q_tiles = seq_len_q // block_elements
    num_kv_tiles = seq_len_kv // block_elements
    if q_variable_block_sizes is None:
        if num_q_tiles == variable_block_sizes.numel():
            q_variable_block_sizes = variable_block_sizes
        else:
            raise ValueError(
                "q_variable_block_sizes must be provided when q and kv have different "
                f"numbers of tiles (num_q_tiles={num_q_tiles}, num_kv_tiles={num_kv_tiles})."
            )
    else:
        assert q_variable_block_sizes.numel() == num_q_tiles, (
            f"q_variable_block_sizes must have length seq_len_q//{block_elements}, "
            f"but got {q_variable_block_sizes.numel()} vs {num_q_tiles}"
        )

    q_compress = (
        q.view(batch_size, num_heads, num_q_tiles, block_elements, head_dim)
        .float()
        .sum(dim=3)
        / q_variable_block_sizes.view(1, 1, -1, 1)
    ).to(q.dtype)
    k_compress = (
        k.view(batch_size, num_heads, num_kv_tiles, block_elements, head_dim)
        .float()
        .sum(dim=3)
        / variable_block_sizes.view(1, 1, -1, 1)
    ).to(k.dtype)
    v_compress = (
        v.view(batch_size, num_heads, num_kv_tiles, block_elements, head_dim)
        .float()
        .sum(dim=3)
        / variable_block_sizes.view(1, 1, -1, 1)
    ).to(v.dtype)

    output_compress, block_attn_score = torch_attention(q_compress, k_compress,
                                                        v_compress)

    output_compress = output_compress.view(batch_size, num_heads,
                                           seq_len_q // block_elements, 1,
                                           head_dim)
    output_compress = output_compress.repeat(1, 1, 1, block_elements,
                                             1).view(batch_size, num_heads,
                                                     seq_len_q, head_dim)

    topK_indices = torch.topk(block_attn_score, topk, dim=-1).indices
    block_mask = torch.zeros_like(block_attn_score, dtype=torch.bool).scatter_(-1, topK_indices, True)
    output_select, _ = block_sparse_attn(q, k, v, block_mask, variable_block_sizes)

    if compress_attn_weight is not None:
        final_output = output_compress * compress_attn_weight + output_select
    else:
        final_output = output_compress + output_select

    # -----------------------------
    # Optional debug visualization
    # -----------------------------
    # Enable via:
    #   FASTVIDEO_VSA_DEBUG_SAVE=1
    # Optionally set:
    #   FASTVIDEO_VSA_DEBUG_DIR=/path/to/save
    #   FASTVIDEO_VSA_DEBUG_PREFIX=some_tag (e.g., validation_step_50_prompt_hash)
    #   FASTVIDEO_VSA_DEBUG_MAX_SAVES=200
    if os.environ.get("FASTVIDEO_VSA_DEBUG_SAVE", "0") == "1":
        # Throttle saves to avoid blowing up disk.
        max_saves = int(os.environ.get("FASTVIDEO_VSA_DEBUG_MAX_SAVES", "200"))
        once_per_block = os.environ.get("FASTVIDEO_VSA_DEBUG_ONCE_PER_BLOCK",
                                        "1") == "1"

        # Global process-local counter.
        if not hasattr(video_sparse_attn, "_dbg_save_count"):
            video_sparse_attn._dbg_save_count = 0  # type: ignore[attr-defined]
        if not hasattr(video_sparse_attn, "_dbg_seen_prefix_block"):
            video_sparse_attn._dbg_seen_prefix_block = set()  # type: ignore[attr-defined]

        base_dir = Path(os.environ.get("FASTVIDEO_VSA_DEBUG_DIR", "vsa_debug"))
        prefix = os.environ.get("FASTVIDEO_VSA_DEBUG_PREFIX", "default")

        # Infer "tiles per causal block" from Q tiles (causal path uses one Q-block).
        tiles_per_block = int(num_q_tiles)
        if tiles_per_block <= 0:
            raise RuntimeError(
                "FASTVIDEO_VSA_DEBUG_SAVE=1 but inferred tiles_per_block<=0. "
                f"num_q_tiles={num_q_tiles} (seq_len_q={seq_len_q})"
            )

        if (num_kv_tiles % tiles_per_block) == 0:
            kv_blocks = num_kv_tiles // tiles_per_block
        else:
            kv_blocks = 1

        # Determine which block index to use for dedup / filenames.
        # By default, we infer it from (kv_tiles / q_tiles) assuming "one Q block" causal calls.
        # For causal VSA with sliding-window KV cache, this inferred value can stay small even
        # as the *absolute* causal block index grows. In that case, the caller can provide:
        #   FASTVIDEO_VSA_DEBUG_BLOCK_IDX=<absolute_block_idx>
        # and we will use that for naming/dedup.
        env_block_idx = os.environ.get("FASTVIDEO_VSA_DEBUG_BLOCK_IDX", None)
        if env_block_idx is not None:
            try:
                kv_block_idx = int(env_block_idx)
            except Exception as e:
                raise RuntimeError(
                    "FASTVIDEO_VSA_DEBUG_SAVE=1 but FASTVIDEO_VSA_DEBUG_BLOCK_IDX "
                    f"is not a valid int: {env_block_idx!r}"
                ) from e
        else:
            # For causal generation, the current KV block index is the last block in the prefix (window).
            kv_block_idx = max(0, kv_blocks - 1)

        # Dedup early (do NOT consume the save budget for repeats).
        if once_per_block:
            seen_key = (prefix, kv_block_idx)
            if seen_key in video_sparse_attn._dbg_seen_prefix_block:  # type: ignore[attr-defined]
                return final_output

        # Throttle based on the number of *actual PNG writes* (not number of calls).
        if video_sparse_attn._dbg_save_count >= max_saves:  # type: ignore[attr-defined]
            return final_output
        video_sparse_attn._dbg_save_count += 1  # type: ignore[attr-defined]

        # Mark as seen only when we will actually write an image.
        if once_per_block:
            video_sparse_attn._dbg_seen_prefix_block.add(seen_key)  # type: ignore[attr-defined]

        base_dir.mkdir(parents=True, exist_ok=True)
        out_dir = base_dir / prefix
        out_dir.mkdir(parents=True, exist_ok=True)

        # Compress attention distribution over KV tiles from softmax scores.
        # block_attn_score: [B, H, q_tiles, kv_tiles]  (softmax over kv_tiles)
        attn_over_kv = block_attn_score.float().mean(dim=(0, 1, 2))  # [kv_tiles]

        # Sparse selection statistics from topk indices.
        # topK_indices: [B, H, q_tiles, topk] in kv-tile space.
        sel_idx = topK_indices.reshape(-1).to(torch.long)  # [B*H*q_tiles*topk]
        sel_counts = torch.bincount(sel_idx, minlength=num_kv_tiles).float()  # total picks per kv-tile
        sel_unique = (sel_counts > 0).float()  # whether a kv-tile was ever selected (k-block count proxy)

        # Aggregate by KV block id
        def _agg_by_block(vec: torch.Tensor) -> torch.Tensor:
            if kv_blocks == 1:
                return vec.sum().view(1)
            vec = vec.view(kv_blocks, tiles_per_block).sum(dim=1)
            return vec

        attn_by_block = _agg_by_block(attn_over_kv)
        # Total selection mass (counts) and unique selected tile counts per KV block.
        sel_by_block = _agg_by_block(sel_counts)
        sel_unique_by_block = _agg_by_block(sel_unique)
        attn_by_block = attn_by_block / (attn_by_block.sum() + 1e-9)
        sel_by_block = sel_by_block / (sel_by_block.sum() + 1e-9)

        # Build a simple heatmap: q_tile x kv_block (compress attention mass)
        heat = block_attn_score.float().mean(dim=(0, 1))  # [q_tiles, kv_tiles]
        if kv_blocks > 1:
            heat = heat.view(num_q_tiles, kv_blocks,
                             tiles_per_block).sum(dim=2)  # [q_tiles, kv_blocks]
        else:
            heat = heat.sum(dim=1, keepdim=True)  # [q_tiles, 1]
        # Normalize for visualization (row-wise)
        heat = heat / (heat.sum(dim=1, keepdim=True) + 1e-9)

        # Save as PNG (PIL-based). Do not silently skip failures.
        import numpy as np
        from PIL import Image, ImageDraw

        def _bar_plot(values: np.ndarray, width: int = 480,
                      height: int = 120) -> Image.Image:
            values = np.clip(values, 0.0, 1.0)
            n = max(1, values.shape[0])
            img = Image.new("RGB", (width, height), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            pad = 6
            bw = max(1, (width - 2 * pad) // n)
            for i, v_ in enumerate(values):
                x0 = pad + i * bw
                x1 = x0 + bw - 1
                y1 = height - pad
                y0 = int(y1 - v_ * (height - 2 * pad))
                draw.rectangle([x0, y0, x1, y1], fill=(60, 120, 220))
            return img

        def _heatmap(mat: np.ndarray, scale: int = 6) -> Image.Image:
            """
            Render q_tile x kv_block heatmap.

            By default we output a *square* heatmap (Q axis squashed) for readability, since
            q_tiles can be large (e.g. 104) while kv_blocks is small (e.g. 6).
            Control via:
              - FASTVIDEO_VSA_DEBUG_HEATMAP_SQUARE=1/0
              - FASTVIDEO_VSA_DEBUG_HEATMAP_SIDE=360  (only used when square=1)
              - FASTVIDEO_VSA_DEBUG_HEATMAP_CMAP=turbo|gray  (default: turbo)
            """
            mat = np.clip(mat, 0.0, 1.0)
            h, w = mat.shape

            cmap = os.environ.get("FASTVIDEO_VSA_DEBUG_HEATMAP_CMAP",
                                  "turbo").lower()

            def _turbo_rgb(x01: np.ndarray) -> np.ndarray:
                """
                Turbo colormap approximation (Google).
                Input: x in [0,1], shape [H,W]
                Output: uint8 RGB, shape [H,W,3]
                """
                x = x01.astype(np.float32)
                # Coeffs from turbo colormap polynomial approximation.
                r = (0.13572138 + x * (4.61539260 + x *
                                      (-42.66032258 + x *
                                       (132.13108234 + x *
                                        (-152.94239396 + x * 59.28637943)))))
                g = (0.09140261 + x * (2.19418839 + x *
                                      (4.84296658 + x *
                                       (-14.18503333 + x *
                                        (4.27729857 + x * 2.82956604)))))
                b = (0.10667330 + x * (12.64194608 + x *
                                      (-60.58204836 + x *
                                       (110.36276771 + x *
                                        (-89.90310912 + x * 27.34824973)))))
                rgb = np.stack([r, g, b], axis=-1)
                rgb = np.clip(rgb, 0.0, 1.0)
                return (rgb * 255.0).astype(np.uint8)

            if cmap == "gray" or cmap == "greyscale" or cmap == "grayscale":
                arr = (mat * 255).astype(np.uint8)
                img = Image.fromarray(arr, mode="L").convert("RGB")
            elif cmap == "turbo":
                rgb = _turbo_rgb(mat)
                img = Image.fromarray(rgb, mode="RGB")
            else:
                raise RuntimeError(
                    "FASTVIDEO_VSA_DEBUG_HEATMAP_CMAP must be 'turbo' or 'gray', "
                    f"but got {cmap!r}")

            square = os.environ.get("FASTVIDEO_VSA_DEBUG_HEATMAP_SQUARE",
                                    "1") == "1"
            if square:
                try:
                    side = int(os.environ.get("FASTVIDEO_VSA_DEBUG_HEATMAP_SIDE",
                                              "360"))
                except Exception as e:
                    raise RuntimeError(
                        "FASTVIDEO_VSA_DEBUG_HEATMAP_SIDE must be an int") from e
                side = max(32, side)
                img = img.resize((side, side), resample=Image.NEAREST)
            else:
                img = img.resize((w * scale, h * scale),
                                 resample=Image.NEAREST)
            return img

        attn_img = _bar_plot(attn_by_block.cpu().numpy())

        # Panel 3: selection heatmap (q_tile x kv_tile) derived from topK indices.
        # This answers: "for each q_tile, which kv_tiles were selected?"
        # We aggregate selections across (B, H, topk) into counts, then normalize row-wise.
        # NOTE: This is debug-only; a small Python loop over q_tiles is acceptable.
        sel_heat = torch.zeros((num_q_tiles, num_kv_tiles),
                               device=block_attn_score.device,
                               dtype=torch.float32)
        # [q_tiles, B*H*topk]
        idx_q = topK_indices.permute(2, 0, 1, 3).reshape(num_q_tiles, -1).to(
            torch.long)
        for qi in range(num_q_tiles):
            counts = torch.bincount(idx_q[qi], minlength=num_kv_tiles).float()
            sel_heat[qi] = counts
        sel_heat = sel_heat / (sel_heat.sum(dim=1, keepdim=True) + 1e-9)
        # Make the selection heatmap more contrast-sensitive:
        # - Use robust percentile stretch on the *values* to avoid "all one color"
        # - Optional log/gamma to enhance small variations
        #
        # Controls:
        #   FASTVIDEO_VSA_DEBUG_SEL_HEATMAP_PCT_LOW=1
        #   FASTVIDEO_VSA_DEBUG_SEL_HEATMAP_PCT_HIGH=99
        #   FASTVIDEO_VSA_DEBUG_SEL_HEATMAP_GAMMA=1.0  (set <1 to boost low values)
        #   FASTVIDEO_VSA_DEBUG_SEL_HEATMAP_LOG=0/1
        sel_np = sel_heat.cpu().numpy().astype(np.float32)
        if os.environ.get("FASTVIDEO_VSA_DEBUG_SEL_HEATMAP_LOG", "0") == "1":
            # log1p keeps zeros at zero and expands small values.
            sel_np = np.log1p(sel_np)
        try:
            pct_low = float(os.environ.get("FASTVIDEO_VSA_DEBUG_SEL_HEATMAP_PCT_LOW", "1"))
            pct_high = float(os.environ.get("FASTVIDEO_VSA_DEBUG_SEL_HEATMAP_PCT_HIGH", "99"))
        except Exception as e:
            raise RuntimeError(
                "FASTVIDEO_VSA_DEBUG_SEL_HEATMAP_PCT_LOW/HIGH must be floats"
            ) from e
        vmin = float(np.percentile(sel_np, pct_low))
        vmax = float(np.percentile(sel_np, pct_high))
        if vmax <= vmin:
            vmax = vmin + 1e-9
        sel_np = np.clip((sel_np - vmin) / (vmax - vmin), 0.0, 1.0)
        try:
            gamma = float(os.environ.get("FASTVIDEO_VSA_DEBUG_SEL_HEATMAP_GAMMA", "1.0"))
        except Exception as e:
            raise RuntimeError("FASTVIDEO_VSA_DEBUG_SEL_HEATMAP_GAMMA must be a float") from e
        if gamma != 1.0:
            gamma = max(1e-6, gamma)
            sel_np = np.power(sel_np, gamma)
        sel_heat_img = _heatmap(sel_np)
        heat_img = _heatmap(heat.cpu().numpy())

        # Stack: heatmap on top, then bar plot, then selection heatmap
        W = max(heat_img.width, attn_img.width, sel_heat_img.width)
        H = heat_img.height + attn_img.height + sel_heat_img.height + 40
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # Title
        title = (
            f"VSA debug: q_tiles={num_q_tiles} kv_tiles={num_kv_tiles} "
            f"kv_blocks(window)={kv_blocks} kv_block_idx={kv_block_idx} topk={topk}"
        )
        draw.text((10, 5), title, fill=(0, 0, 0))

        y = 30
        canvas.paste(heat_img, (10, y))
        y += heat_img.height + 10
        draw.text((10, y), "compress attn mass per KV block", fill=(0, 0, 0))
        y += 16
        canvas.paste(attn_img, (10, y))
        y += attn_img.height + 10
        draw.text(
            (10, y),
            "topk selection heatmap (q_tile x kv_tile, row-normalized; contrast-stretched)",
            fill=(0, 0, 0),
        )
        y += 16
        canvas.paste(sel_heat_img, (10, y))

        ts = int(time.time() * 1000)
        tag = md5(
            f"{ts}-{video_sparse_attn._dbg_save_count}".encode()  # type: ignore[attr-defined]
        ).hexdigest()[:8]
        out_path = out_dir / f"vsa_b{kv_block_idx:03d}_{video_sparse_attn._dbg_save_count:05d}_{tag}.png"  # type: ignore[attr-defined]
        canvas.save(out_path)
    return final_output

