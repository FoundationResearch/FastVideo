import torch
import torch.nn as nn

from typing import Dict, List, Tuple

from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.logger import init_logger

from .SVDQuantLinearManual import SVDQuantLinearManual

logger = init_logger(__name__)


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
    input_map: Dict[str, torch.Tensor] | None = None,
) -> nn.Module:
    """Replace all ReplicatedLinear modules in-place with SVDQuantReplicatedLinear.

    If input_map is provided, it should map qualified module names to representative
    pre-activation inputs for computing smooth factors during calibration.
    """

    # Global warning if no calibration inputs are provided
    if input_map is None or len(input_map) == 0:
        logger.warning(
            "SVDQuant: No calibration inputs were provided; performing uncalibrated replacement (smooth=1) for all layers."
        )

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

            # If calibration inputs available for this layer, compute calibrated SVDQ
            layer_inputs = input_map.get(name) if input_map is not None else None
            if layer_inputs is not None:
                adapter.calibrate_and_load_from_replicated(
                    module,
                    layer_inputs=layer_inputs,
                    rank=rank,
                    w_percentile=w_percentile,
                    act_unsigned=act_unsigned,
                )
            else:
                logger.warning(
                    "SVDQuant: Layer %s has no calibration inputs; using uncalibrated replacement (smooth=1).",
                    name,
                )
                adapter.load_from_replicated(module)

            # Verbose: report linear shape, chosen rank, and max(smooth)
            n, k = module.weight.shape  # type: ignore[attr-defined]
            r = int(adapter.svdq._manual_lora_down.shape[1])
            smax = float(adapter.svdq._manual_smooth.max().item())
            logger.info(
                "SVDQuant: %s weight=(%d,%d) -> lora_rank=%d + int4(gs=%d), max(smooth)=%.4f (calibrated=%s)",
                name, n, k, r, adapter.svdq.group_size, smax, str(layer_inputs is not None),
            )
            to_replace.append((parent, child_name, adapter))

    for parent, child_name, adapter in to_replace:
        setattr(parent, child_name, adapter)

    return model


@torch.no_grad()
def register_replicated_linear_input_hooks(
    model: nn.Module,
    *,
    max_rows_per_layer: int = 8192,
) -> tuple[List[torch.utils.hooks.RemovableHandle], Dict[str, torch.Tensor]]:
    """Register pre-forward hooks on all ReplicatedLinear layers to collect inputs.

    Returns hook handles and a dict mapping qualified module names to collected inputs.
    """
    buffers: Dict[str, torch.Tensor] = {}
    handles: list = []

    def make_hook(qualified_name: str):
        def hook(mod, inp):
            try:
                x = inp[0]
                if isinstance(x, tuple) or isinstance(x, list):
                    x = x[0]
                if x.ndim == 3:
                    x = x.reshape(-1, x.shape[-1])
                elif x.ndim == 2:
                    pass
                else:
                    x = x.view(-1, x.shape[-1])
                x = x.detach()
                if qualified_name in buffers:
                    old = buffers[qualified_name]
                    need = max_rows_per_layer - old.shape[0]
                    if need <= 0:
                        return
                    take = min(need, x.shape[0])
                    if take > 0:
                        # Move slice to CPU to avoid GPU memory growth during calibration
                        buffers[qualified_name] = torch.cat(
                            [old, x[:take].to(device="cpu", non_blocking=True)], dim=0)
                else:
                    # Store on CPU to minimize GPU memory usage
                    buffers[qualified_name] = x[:max_rows_per_layer].to(device="cpu", non_blocking=True)
            except Exception:
                # Be robust; skip on any odd-shaped input
                return
        return hook

    for name, module in model.named_modules():
        if isinstance(module, ReplicatedLinear):  # type: ignore[arg-type]
            handles.append(module.register_forward_pre_hook(make_hook(name)))

    return handles, buffers


@torch.no_grad()
def _build_calibrated_svdq_from_replicated(
    layer: "ReplicatedLinear",  # type: ignore[name-defined]
    layer_inputs: torch.Tensor,
    *,
    rank: int,
    w_percentile: float | None,
    act_unsigned: bool,
    ) -> SVDQuantLinearManual:
    """Utility to create a calibrated SVDQuantLinearManual from a ReplicatedLinear layer."""
    device = next(layer.parameters()).device
    dtype = getattr(layer, "params_dtype", torch.bfloat16)
    lin = nn.Linear(layer.input_size, layer.output_size, bias=(layer.bias is not None), device=device, dtype=dtype)
    with torch.no_grad():
        lin.weight.copy_(layer.weight.to(dtype))  # type: ignore[attr-defined]
        if layer.bias is not None:
            lin.bias.copy_(layer.bias.to(dtype))  # type: ignore[attr-defined]
    return SVDQuantLinearManual.from_linear_and_inputs(
        lin,
        layer_inputs,
        rank=rank,
        w_percentile=w_percentile,
        act_unsigned=act_unsigned,
        skip_norm_clamp=True,
    )


def _ensure_device_dtype(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if t.device != ref.device or t.dtype != ref.dtype:
        return t.to(device=ref.device, dtype=ref.dtype)
    return t


def _maybe_downsample_inputs(x: torch.Tensor, max_rows: int = 4096) -> torch.Tensor:
    if x.shape[0] > max_rows:
        return x[:: (x.shape[0] // max_rows + 1)]
    return x


# Extend SVDQuantReplicatedLinear with a calibration-aware loader
def _svdqrl_calibrate_and_load_from_replicated(
    self: SVDQuantReplicatedLinear,
    layer: "ReplicatedLinear",  # type: ignore[name-defined]
    *,
    layer_inputs: torch.Tensor,
    rank: int,
    w_percentile: float | None,
    act_unsigned: bool,
) -> None:
    x = _maybe_downsample_inputs(layer_inputs)
    x = _ensure_device_dtype(x, next(layer.parameters()))
    svdq_mod = _build_calibrated_svdq_from_replicated(
        layer,
        x,
        rank=rank,
        w_percentile=w_percentile,
        act_unsigned=act_unsigned,
    )
    self.svdq = svdq_mod
    # keep a copy of original weight for compatibility (dtype/shape checks)
    self.weight.copy_(layer.weight.to(self.weight.dtype))  # type: ignore[attr-defined]
    if layer.bias is not None:
        self.svdq.bias.copy_(layer.bias.to(self.svdq.dtype))  # type: ignore[attr-defined]


# bind method to class
setattr(SVDQuantReplicatedLinear, "calibrate_and_load_from_replicated", _svdqrl_calibrate_and_load_from_replicated)


