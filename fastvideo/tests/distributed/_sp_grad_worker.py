# SPDX-License-Identifier: Apache-2.0
"""
Worker entrypoint for SP gradient parity tests.

This module is intended to be executed under torchrun, e.g.
  torchrun --standalone --nproc_per_node=2 -m fastvideo.tests.distributed._sp_grad_worker

Environment variables:
  - SP_SIZE: sequence parallel size to initialize (int)
  - OUT_FILE: path to write rank0 gradient snapshot (str)
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from types import SimpleNamespace

import torch
import torch.distributed as dist

from fastvideo.distributed import (
    cleanup_dist_env_and_memory,
    get_dp_group,
    get_sp_group,
    maybe_init_distributed_environment_and_model_parallel,
)
from fastvideo.pipelines import TrainingBatch
from fastvideo.training.training_pipeline import TrainingPipeline


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _broadcast_(t: torch.Tensor, src: int = 0) -> None:
    dist.broadcast(t, src=src)


class DummyTransformer(torch.nn.Module):
    """
    Minimal transformer-like module matching TrainingPipeline input kwargs.

    It returns a tensor shaped like hidden_states so TrainingPipeline can treat
    it as model_pred directly.
    """

    def __init__(self, c: int, init: torch.Tensor):
        super().__init__()
        assert init.shape == (c,)
        self.scale = torch.nn.Parameter(init.clone())

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        timestep=None,
        encoder_attention_mask=None,
        return_dict: bool = False,
    ) -> torch.Tensor:
        _ = (encoder_hidden_states, timestep, encoder_attention_mask, return_dict)
        return hidden_states * self.scale.view(1, -1, 1, 1, 1)


class _MinimalTrainingPipeline(TrainingPipeline):
    # We never call TrainingPipeline.__init__ in this test worker, but we must
    # implement the abstract method to make the class concrete.
    def initialize_validation_pipeline(self, training_args):  # type: ignore[override]
        raise NotImplementedError


def _make_minimal_training_pipeline(transformer: torch.nn.Module):
    """
    Create a TrainingPipeline instance without running its heavy __init__.

    We only need a few attributes/methods for _build_input_kwargs() and
    _transformer_forward_and_compute_loss().
    """
    pipe = _MinimalTrainingPipeline.__new__(_MinimalTrainingPipeline)
    pipe.transformer = transformer
    pipe.transformer_2 = None
    pipe.train_transformer_2 = False
    pipe.tracker = SimpleNamespace(timed=lambda *_args, **_kwargs: nullcontext())
    pipe.training_args = SimpleNamespace(
        precondition_outputs=False,
        gradient_accumulation_steps=1,
    )
    return pipe


def _average_grads_like_training_stack_(grad: torch.Tensor) -> torch.Tensor:
    """
    Mimic the expected gradient reductions:
      - DP (replicate) averages gradients across dp_group
      - SP (sequence parallel) averages gradients across sp_group

    In FastVideo, the actual reduction is handled by the wrapping strategy
    (DDP/FSDP/HSDP). Here we explicitly reduce so this worker can run with a
    plain nn.Module.
    """
    grad = grad.clone()

    dp = get_dp_group()
    if dp.world_size > 1:
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=dp.device_group)
        grad /= dp.world_size

    sp = get_sp_group()
    if sp.world_size > 1:
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=sp.device_group)
        grad /= sp.world_size

    return grad


def main() -> None:
    sp_size = int(os.environ["SP_SIZE"])
    out_file = os.environ["OUT_FILE"]

    if not torch.cuda.is_available():
        raise RuntimeError("This worker requires CUDA.")

    torch.cuda.set_device(_local_rank())

    # Initialize FastVideo dist + model-parallel groups.
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=sp_size)

    device = torch.device(f"cuda:{_local_rank()}")
    torch.manual_seed(2029)

    # Choose a shape that triggers SP padding for sp_size=2 (thw odd).
    b, c, t, h, w = 2, 8, 1, 1, 9

    noisy = torch.empty((b, c, t, h, w), device=device, dtype=torch.float32)
    latents = torch.empty((b, c, t, h, w), device=device, dtype=torch.float32)
    noise = torch.empty((b, c, t, h, w), device=device, dtype=torch.float32)
    init = torch.empty((c,), device=device, dtype=torch.float32)

    if _rank() == 0:
        noisy.normal_()
        latents.normal_()
        noise.normal_()
        init.normal_()
    _broadcast_(noisy)
    _broadcast_(latents)
    _broadcast_(noise)
    _broadcast_(init)

    transformer = DummyTransformer(c=c, init=init).to(device)
    pipe = _make_minimal_training_pipeline(transformer)

    batch = TrainingBatch()
    batch.current_timestep = 0
    batch.noisy_model_input = noisy
    batch.latents = latents
    batch.noise = noise
    batch.timesteps = torch.tensor([1], device=device, dtype=torch.int64)
    batch.encoder_hidden_states = None
    batch.encoder_attention_mask = None
    batch.attn_metadata = None
    batch.total_loss = 0.0

    # Build kwargs using the real TrainingPipeline helper.
    batch = TrainingPipeline._build_input_kwargs(pipe, batch)  # type: ignore[misc]
    # Run the real TrainingPipeline forward+loss+backward code path.
    _ = TrainingPipeline._transformer_forward_and_compute_loss(pipe, batch)  # type: ignore[misc]

    grad = transformer.scale.grad
    assert grad is not None
    grad = _average_grads_like_training_stack_(grad.detach())

    if _rank() == 0:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        torch.save(
            {
                "scale_grad": grad.cpu(),
                # This is the reduced (world avg) loss accumulated in TrainingPipeline.
                # For this worker we run exactly 1 step, so total_loss == avg_loss.
                "total_loss": float(batch.total_loss or 0.0),
            },
            out_file,
        )

    dist.barrier()
    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()


