# SPDX-License-Identifier: Apache-2.0
"""Attention map capture hook for WanVideo models.

Supports Flash Attention and VSA backends by computing attention weights
manually from Q, K tensors (bypassing fused kernels that don't expose
intermediate weights).

Usage::

    from fastvideo.attention.attn_map import (
        AttentionMapStore, attach_attention_map_hooks,
        detach_attention_map_hooks,
    )

    store = attach_attention_map_hooks(model)

    with torch.no_grad():
        output = pipeline(...)

    # store.maps: dict[layer_name -> list of Tensors]
    # Each tensor shape: [B, S_q, S_k] (head-averaged)
    #                 or [B, H, S_q, S_k] (head_reduction="none")

    from fastvideo.attention.visualization import plot_attention_maps
    plot_attention_maps(store, dit_seq_shape=(T, H, W))
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.hooks.hooks import ForwardHook, ModuleHookManager
from fastvideo.layers.rotary_embedding import _apply_rotary_emb
from fastvideo.logger import init_logger

logger = init_logger(__name__)


class AttentionMapStore:
    """Thread-safe store for captured attention weight maps per layer.

    Maps are kept on CPU to avoid GPU OOM during long inference runs.
    Call ``clear()`` between inference steps when only the final step's
    maps are of interest.
    """

    def __init__(self) -> None:
        # layer_name -> list[Tensor]  (one entry per forward call)
        self.maps: dict[str, list[torch.Tensor]] = defaultdict(list)

    def add(self, layer_name: str, attn_weights: torch.Tensor) -> None:
        self.maps[layer_name].append(attn_weights.detach().cpu())

    def clear(self) -> None:
        self.maps.clear()

    def get(self, layer_name: str) -> list[torch.Tensor]:
        return self.maps.get(layer_name, [])

    def layer_names(self) -> list[str]:
        return list(self.maps.keys())

    def __repr__(self) -> str:
        info = {k: len(v) for k, v in self.maps.items()}
        return f"AttentionMapStore({info})"


class AttentionMapHook(ForwardHook):
    """Captures attention weight maps from DistributedAttention layers.

    Works with both standard (Flash Attention) and VSA-based
    (DistributedAttention_VSA) layers by computing softmax(QK^T/√D)
    manually from the captured Q, K tensors.

    In a sequence-parallel setup the hook performs an all-gather across
    SP ranks before computing weights so each rank stores the full-sequence
    map.  On a single GPU the gather is a no-op.

    Args:
        layer_name: Identifier used as the key in ``AttentionMapStore``.
        store: Destination store for computed maps.
        head_reduction: ``"mean"`` (default) averages over heads; ``"none"``
            keeps all heads in the output.
        max_seq_len: If the sequence length exceeds this value, tokens are
            uniformly sub-sampled before computing the attention matrix to
            keep memory manageable.
    """

    _HOOK_NAME = "attention_map"

    def __init__(
        self,
        layer_name: str,
        store: AttentionMapStore,
        head_reduction: str = "mean",
        max_seq_len: int = 2048,
    ) -> None:
        self.layer_name = layer_name
        self.store = store
        self.head_reduction = head_reduction
        self.max_seq_len = max_seq_len

        # Temporaries populated by pre_forward, consumed by post_forward.
        self._q: torch.Tensor | None = None
        self._k: torch.Tensor | None = None
        self._freqs_cis: tuple[torch.Tensor, torch.Tensor] | None = None
        self._original_seq_len: int | None = None
        self._softmax_scale: float = 1.0

    @classmethod
    def name(cls) -> str:
        return cls._HOOK_NAME

    def on_attach(self, module: nn.Module) -> None:
        self._softmax_scale = float(getattr(module, "softmax_scale", 1.0))

    # ------------------------------------------------------------------
    # Hook interface
    # ------------------------------------------------------------------

    def pre_forward(
        self,
        module: nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        # DistributedAttention.forward(q, k, v, original_seq_len, ...,
        #                              freqs_cis=(...))
        # DistributedAttention_VSA.forward(..., gate_compress=...)
        if len(args) >= 2:
            self._q = args[0].detach().float()
            self._k = args[1].detach().float()
        else:
            self._q = self._k = None

        # original_seq_len may be positional arg[3] or keyword
        if len(args) >= 4 and isinstance(args[3], int):
            self._original_seq_len = args[3]
        else:
            self._original_seq_len = kwargs.get("original_seq_len", None)

        self._freqs_cis = kwargs.get("freqs_cis", None)
        return args, kwargs

    def post_forward(self, module: nn.Module, output: Any) -> Any:
        if self._q is None or self._k is None:
            return output
        try:
            attn_weights = self._compute_attn_weights()
            if attn_weights is not None:
                self.store.add(self.layer_name, attn_weights)
        except Exception as exc:
            warnings.warn(
                f"AttentionMapHook: failed to compute weights for "
                f"'{self.layer_name}': {exc}",
                stacklevel=2,
            )
        finally:
            self._q = None
            self._k = None
            self._freqs_cis = None
            self._original_seq_len = None
        return output

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_attn_weights(self) -> torch.Tensor | None:
        q = self._q  # [B, local_S, H, D]
        k = self._k

        # Gather full sequence across SP ranks (no-op on single GPU).
        q = self._sp_all_gather(q)  # [B, full_S, H, D]
        k = self._sp_all_gather(k)

        # Trim padding added by SP all-to-all.
        if self._original_seq_len is not None:
            q = q[:, :self._original_seq_len]
            k = k[:, :self._original_seq_len]

        # Apply rotary positional embeddings.
        if self._freqs_cis is not None:
            try:
                cos, sin = self._freqs_cis
                q = _apply_rotary_emb(q, cos, sin, is_neox_style=False)
                k = _apply_rotary_emb(k, cos, sin, is_neox_style=False)
            except Exception as exc:
                warnings.warn(
                    f"AttentionMapHook: could not apply RoPE for "
                    f"'{self.layer_name}' (continuing without RoPE): {exc}",
                    stacklevel=2,
                )

        # Sub-sample if sequence is too long to visualize comfortably.
        S = q.shape[1]
        if S > self.max_seq_len:
            step = max(1, S // self.max_seq_len)
            q = q[:, ::step]
            k = k[:, ::step]

        # Compute attention weights: [B, H, S_q, S_k]
        # q, k: [B, S, H, D] -> [B, H, S, D]
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * self._softmax_scale
        attn_weights = F.softmax(scores, dim=-1)  # [B, H, S_q, S_k]

        if self.head_reduction == "mean":
            attn_weights = attn_weights.mean(dim=1)  # [B, S_q, S_k]

        return attn_weights

    @staticmethod
    def _sp_all_gather(x: torch.Tensor) -> torch.Tensor:
        """All-gather along the sequence dimension across SP ranks."""
        try:
            import torch.distributed as dist
            from fastvideo.distributed.parallel_state import get_sp_group
            sp_group = get_sp_group()
            if sp_group.world_size <= 1:
                return x
            parts = [torch.empty_like(x) for _ in range(sp_group.world_size)]
            dist.all_gather(parts, x.contiguous(),
                            group=sp_group.device_group)
            return torch.cat(parts, dim=1)
        except Exception:
            return x


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------


def attach_attention_map_hooks(
    model: nn.Module,
    store: AttentionMapStore | None = None,
    layer_names: list[str] | None = None,
    head_reduction: str = "mean",
    max_seq_len: int = 2048,
    self_attn_only: bool = True,
) -> AttentionMapStore:
    """Attach attention-map hooks to WanVideo transformer blocks.

    Hooks are attached to the ``attn1`` (self-attention) sub-module of every
    ``WanTransformerBlock`` and ``WanTransformerBlock_VSA`` found inside
    *model*.  Pass ``self_attn_only=False`` to also hook the ``attn``
    (cross-attention) ``LocalAttention`` layers.

    Args:
        model: The WanVideo model (or any module containing WanTransformerBlock
            sub-modules).
        store: Existing ``AttentionMapStore`` to reuse; a new one is created if
            *None*.
        layer_names: If given, only attach to layers whose generated name is in
            this list.  Useful for targeting specific transformer blocks.
        head_reduction: Passed to ``AttentionMapHook``.
        max_seq_len: Passed to ``AttentionMapHook``.
        self_attn_only: When ``True``, only hook distributed self-attention
            (``attn1``).  When ``False``, also hook cross-attention layers.

    Returns:
        The ``AttentionMapStore`` that will accumulate maps during inference.
    """
    from fastvideo.attention.layer import (DistributedAttention,
                                           DistributedAttention_VSA,
                                           LocalAttention)

    if store is None:
        store = AttentionMapStore()

    hooked: list[str] = []
    for module_name, module in model.named_modules():
        # Target distributed self-attention layers (attn1).
        is_dist_attn = isinstance(module,
                                  (DistributedAttention, DistributedAttention_VSA))
        if self_attn_only and is_dist_attn:
            _try_attach(module, module_name, store, head_reduction, max_seq_len,
                        layer_names, hooked)
        elif not self_attn_only and isinstance(
                module, (DistributedAttention, DistributedAttention_VSA,
                         LocalAttention)):
            _try_attach(module, module_name, store, head_reduction, max_seq_len,
                        layer_names, hooked)

    if not hooked:
        warnings.warn(
            "attach_attention_map_hooks: no DistributedAttention layers found. "
            "Make sure the model contains WanTransformerBlock sub-modules.",
            stacklevel=2,
        )
    else:
        logger.info("Attached attention-map hooks to %d layers: %s",
                    len(hooked), hooked)
    return store


def _try_attach(
    module: nn.Module,
    module_name: str,
    store: AttentionMapStore,
    head_reduction: str,
    max_seq_len: int,
    layer_names: list[str] | None,
    hooked: list[str],
) -> None:
    if layer_names is not None and module_name not in layer_names:
        return
    manager = ModuleHookManager.get_from_or_default(module)
    if manager.get_forward_hook(AttentionMapHook.name()) is not None:
        return  # already attached
    hook = AttentionMapHook(
        layer_name=module_name,
        store=store,
        head_reduction=head_reduction,
        max_seq_len=max_seq_len,
    )
    manager.append_forward_hook(hook)
    hooked.append(module_name)


def detach_attention_map_hooks(model: nn.Module) -> None:
    """Remove all ``AttentionMapHook`` instances from *model*."""
    from fastvideo.attention.layer import (DistributedAttention,
                                           DistributedAttention_VSA,
                                           LocalAttention)

    removed = 0
    for _, module in model.named_modules():
        if not isinstance(module,
                          (DistributedAttention, DistributedAttention_VSA,
                           LocalAttention)):
            continue
        manager = ModuleHookManager.get_from(module)
        if manager is None:
            continue
        if manager.get_forward_hook(AttentionMapHook.name()) is not None:
            manager.remove_forward_hook(AttentionMapHook.name())
            removed += 1

    logger.info("Removed attention-map hooks from %d layers.", removed)
