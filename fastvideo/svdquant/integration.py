import torch
import torch.nn as nn

from typing import Tuple

from fastvideo.layers.linear import ReplicatedLinear

from .SVDQuantLinearManual import SVDQuantLinearManual


class SVDQuantReplicatedLinear(nn.Module):
    """Drop-in replacement for fastvideo.layers.linear.ReplicatedLinear using SVDQuant W4A4.

    It preserves the forward signature by returning (output, output_bias_when_skipped).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        skip_bias_add: bool = False,
        dtype: torch.dtype | None = None,
        rank: int = 32,
        w_percentile: float | None = 0.999,
        act_unsigned: bool = False,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.input_size = in_features
        self.output_size = out_features
        self.skip_bias_add = skip_bias_add
        if dtype is None:
            dtype = torch.bfloat16
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Allocate a placeholder nn.Linear only for constructing SVDQ module
        lin = nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.svdq = SVDQuantLinearManual.from_linear_and_inputs(
            lin,
            None,  # no calibration inputs; use smooth=1
            rank=rank,
            w_percentile=w_percentile,
            act_unsigned=act_unsigned,
        )
        # Expose a compatible weight tensor for downstream dtype/shape queries
        # Note: kept as buffer since we are in inference mode
        self.register_buffer(
            "weight",
            torch.empty(out_features, in_features, dtype=dtype, device=device),
        )

    @torch.no_grad()
    def load_from_replicated(self, layer: "ReplicatedLinear") -> None:  # type: ignore[name-defined]
        # copy weights and bias from the original layer
        assert hasattr(layer, "weight"), "ReplicatedLinear must have attribute weight"
        w = layer.weight.to(self.svdq.dtype)
        self.svdq._manual_smooth.copy_(torch.ones(w.shape[1], dtype=self.svdq.dtype, device=w.device))
        # Rebuild low-rank as 0 and quantize full residual directly
        # Here we reuse from_linear_and_inputs path by setting buffers directly
        # low-rank kept zeros; quantize residual = W_hat (since recon=0)
        residual = w.contiguous()
        N, K = residual.shape
        assert K % self.svdq.group_size == 0, "in_features must be divisible by group_size=64"
        from .SVDQuantLinearManual import _compute_group_scales_sym_int4, _quantize_residual_to_int4
        wscales = _compute_group_scales_sym_int4(residual, group_size=self.svdq.group_size, dtype=self.svdq.dtype, percentile=0.999)
        q_int8 = _quantize_residual_to_int4(residual, wscales, group_size=self.svdq.group_size).to(torch.int8)
        self.svdq._manual_q_int8.copy_(q_int8)
        self.svdq._manual_wscales.copy_(wscales)
        # keep a copy of original weight for compatibility (dtype/shape checks)
        self.weight.copy_(layer.weight.to(self.weight.dtype))
        if layer.bias is not None:
            self.svdq.bias.copy_(layer.bias.to(self.svdq.dtype))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
        # Respect skip_bias_add semantics from ReplicatedLinear
        if self.skip_bias_add:
            saved_bias = getattr(self.svdq, "bias", None)
            if saved_bias is not None:
                self.svdq.bias = None  # type: ignore[attr-defined]
            y = self.svdq(x)
            if saved_bias is not None:
                self.svdq.bias = saved_bias  # type: ignore[attr-defined]
            output_bias = saved_bias
            return y, output_bias
        else:
            y = self.svdq(x)
            return y, None


@torch.no_grad()
def replace_replicated_linear_with_svdq(
    model: nn.Module,
    *,
    rank: int = 32,
    w_percentile: float | None = 0.999,
    act_unsigned: bool = False,
) -> nn.Module:
    """Replace all ReplicatedLinear modules in-place with SVDQuantReplicatedLinear."""

    # Collect replacements with their parents
    to_replace: list[tuple[nn.Module, str, nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, ReplicatedLinear):  # type: ignore[arg-type]
            parent_path = name.split(".")[:-1]
            child_name = name.split(".")[-1]
            parent = model
            for p in parent_path:
                parent = getattr(parent, p)
            # Build adapter with original properties
            adapter = SVDQuantReplicatedLinear(
                in_features=module.input_size,
                out_features=module.output_size,
                bias=(module.bias is not None),
                skip_bias_add=getattr(module, "skip_bias_add", False),
                dtype=getattr(module, "params_dtype", torch.bfloat16),
                rank=rank,
                w_percentile=w_percentile,
                act_unsigned=act_unsigned,
                device=next(module.parameters()).device,
            )
            adapter.load_from_replicated(module)
            to_replace.append((parent, child_name, adapter))

    for parent, child_name, adapter in to_replace:
        setattr(parent, child_name, adapter)

    return model


