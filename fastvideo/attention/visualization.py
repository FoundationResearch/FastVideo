# SPDX-License-Identifier: Apache-2.0
"""Visualization utilities for attention maps captured by
``fastvideo.attention.attn_map.AttentionMapStore``.

All plotting functions use lazy ``matplotlib`` imports so the module
can be imported on headless machines without error.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from fastvideo.attention.attn_map import AttentionMapStore


def plot_attention_maps(
    store: AttentionMapStore,
    timestep_idx: int = -1,
    batch_idx: int = 0,
    dit_seq_shape: tuple[int, int, int] | None = None,
    save_path: str | None = None,
    max_layers: int = 40,
    figsize_per_layer: tuple[float, float] = (6.0, 5.0),
    cmap: str = "viridis",
    show: bool = False,
) -> None:
    """Plot one heatmap per layer showing the full attention matrix.

    Args:
        store: The ``AttentionMapStore`` populated during inference.
        timestep_idx: Which denoising timestep to visualise (index into
            ``store.maps[layer][...]``).  ``-1`` means the last captured
            timestep.
        batch_idx: Batch element to display.
        dit_seq_shape: ``(T, H, W)`` token grid shape.  When provided the
            axes are labelled with spatial coordinates.
        save_path: If given, the figure is saved to this path instead of
            (or in addition to) being shown.
        max_layers: Cap on the number of layers to plot (first N layers).
        figsize_per_layer: ``(width, height)`` of each sub-plot.
        cmap: Matplotlib colour-map name.
        show: Whether to call ``plt.show()``.
    """
    import matplotlib.pyplot as plt

    layer_names = store.layer_names()[:max_layers]
    if not layer_names:
        print("AttentionMapStore is empty -- nothing to plot.")
        return

    n_layers = len(layer_names)
    n_cols = min(4, n_layers)
    n_rows = math.ceil(n_layers / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_layer[0] * n_cols,
                 figsize_per_layer[1] * n_rows),
        squeeze=False,
    )

    for idx, lname in enumerate(layer_names):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        maps = store.get(lname)
        if not maps:
            ax.set_title(f"{lname}\n(no data)")
            ax.axis("off")
            continue

        attn = maps[timestep_idx]  # [B, S_q, S_k] or [B, H, S_q, S_k]
        if attn.dim() == 4:
            attn = attn.mean(dim=1)  # average heads
        attn = attn[batch_idx].float().numpy()  # [S_q, S_k]

        im = ax.imshow(attn, aspect="auto", cmap=cmap,
                        interpolation="nearest")
        # Derive a short label from module path
        short = lname.rsplit(".", 1)[-1] if "." in lname else lname
        ax.set_title(f"L{idx} ({short})", fontsize=9)
        ax.set_xlabel("Key token")
        ax.set_ylabel("Query token")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused axes
    for idx in range(n_layers, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].axis("off")

    fig.suptitle("Layerwise Attention Maps", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved attention-map figure to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_token_attention(
    store: AttentionMapStore,
    query_token_idx: int,
    dit_seq_shape: tuple[int, int, int],
    timestep_idx: int = -1,
    batch_idx: int = 0,
    save_path: str | None = None,
    max_layers: int = 40,
    cmap: str = "hot",
    show: bool = False,
) -> None:
    """For a single query token, show *where* it attends as a spatial map.

    The attention vector (over key tokens) is reshaped to ``(T, H, W)``
    and a mid-frame slice ``(H, W)`` is displayed for each layer.

    Args:
        store: Populated ``AttentionMapStore``.
        query_token_idx: Flat index of the query token to inspect.
        dit_seq_shape: ``(T, H, W)`` token grid shape (after patchification).
        timestep_idx: Denoising timestep index (``-1`` = last).
        batch_idx: Batch element.
        save_path: Optional file path to save the figure.
        max_layers: Maximum layers to plot.
        cmap: Colour map.
        show: Whether to call ``plt.show()``.
    """
    import matplotlib.pyplot as plt

    T, H, W = dit_seq_shape
    layer_names = store.layer_names()[:max_layers]
    if not layer_names:
        print("AttentionMapStore is empty -- nothing to plot.")
        return

    n_layers = len(layer_names)
    n_cols = min(6, n_layers)
    n_rows = math.ceil(n_layers / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.0 * n_cols, 3.0 * n_rows),
        squeeze=False,
    )

    for idx, lname in enumerate(layer_names):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        maps = store.get(lname)
        if not maps:
            ax.axis("off")
            continue

        attn = maps[timestep_idx]
        if attn.dim() == 4:
            attn = attn.mean(dim=1)
        attn = attn[batch_idx].float()  # [S_q, S_k]

        # Clamp query index to actual sub-sampled size.
        S_q, S_k = attn.shape
        q_idx = min(query_token_idx, S_q - 1)
        attn_row = attn[q_idx]  # [S_k]

        # Reshape to spatial grid and take mid-frame.
        if S_k == T * H * W:
            spatial = attn_row.view(T, H, W)
            mid_t = T // 2
            frame = spatial[mid_t].numpy()
        else:
            # Sub-sampled: show as 1-D bar
            frame = attn_row.unsqueeze(0).numpy()

        ax.imshow(frame, cmap=cmap, interpolation="nearest", aspect="auto")
        short = lname.rsplit(".", 1)[-1] if "." in lname else lname
        ax.set_title(f"L{idx} ({short})", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(n_layers, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].axis("off")

    fig.suptitle(
        f"Attention from query token {query_token_idx} "
        f"(mid-frame H x W slice)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved token-attention figure to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_head_attention(
    store: AttentionMapStore,
    layer_name: str,
    timestep_idx: int = -1,
    batch_idx: int = 0,
    save_path: str | None = None,
    cmap: str = "viridis",
    show: bool = False,
) -> None:
    """Plot per-head attention maps for a single layer.

    Only works if the store was populated with ``head_reduction="none"``.

    Args:
        store: Populated ``AttentionMapStore``.
        layer_name: Exact layer name in the store.
        timestep_idx: Denoising timestep index.
        batch_idx: Batch element.
        save_path: Optional save path.
        cmap: Colour map.
        show: Whether to call ``plt.show()``.
    """
    import matplotlib.pyplot as plt

    maps = store.get(layer_name)
    if not maps:
        print(f"No data for layer '{layer_name}'.")
        return

    attn = maps[timestep_idx]
    if attn.dim() != 4:
        print(
            f"Layer '{layer_name}' was captured with head_reduction='mean'. "
            "Re-run with head_reduction='none' to get per-head maps."
        )
        return

    attn = attn[batch_idx].float()  # [H, S_q, S_k]
    n_heads = attn.shape[0]
    n_cols = min(8, n_heads)
    n_rows = math.ceil(n_heads / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.0 * n_cols, 3.0 * n_rows),
        squeeze=False,
    )

    for h in range(n_heads):
        row, col = divmod(h, n_cols)
        ax = axes[row][col]
        ax.imshow(attn[h].numpy(), aspect="auto", cmap=cmap,
                  interpolation="nearest")
        ax.set_title(f"Head {h}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    for h in range(n_heads, n_rows * n_cols):
        row, col = divmod(h, n_cols)
        axes[row][col].axis("off")

    fig.suptitle(f"Per-head attention: {layer_name}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved head-attention figure to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def save_attention_maps(
    store: AttentionMapStore,
    save_path: str,
    timestep_idx: int = -1,
) -> None:
    """Save raw attention weight tensors to a ``.pt`` file.

    Args:
        store: Populated ``AttentionMapStore``.
        save_path: Destination ``.pt`` file path.
        timestep_idx: Which timestep to save (``-1`` = last).
    """
    data: dict[str, torch.Tensor] = {}
    for lname in store.layer_names():
        maps = store.get(lname)
        if maps:
            data[lname] = maps[timestep_idx]

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, save_path)
    print(f"Saved {len(data)} attention maps to {save_path}")
