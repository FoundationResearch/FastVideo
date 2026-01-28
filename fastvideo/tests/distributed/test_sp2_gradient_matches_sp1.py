# SPDX-License-Identifier: Apache-2.0
"""
End-to-end-ish gradient parity test for Sequence Parallel (SP) sharded loss.

Goal:
  When sharding latents across SP into 2 shards (sp_size=2), the resulting
  parameter gradients (after rank-averaging) should match the gradients from
  the "SP disabled" loss computation (full MSE mean) on the same inputs.

Run:
  source /home/hao_lab/miniconda3/etc/profile.d/conda.sh
  conda activate alexfv
  torchrun --standalone --nproc_per_node=2 -m pytest -q fastvideo/tests/distributed/test_sp2_gradient_matches_sp1.py
"""

import os

import pytest
import torch
import torch.distributed as dist

from fastvideo.distributed import cleanup_dist_env_and_memory
from fastvideo.distributed.parallel_state import (
    get_sp_group,
    maybe_init_distributed_environment_and_model_parallel,
)
from fastvideo.training.training_utils import shard_latents_across_sp


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


@pytest.fixture(scope="module")
def dist_setup_sp2():
    ws = _world_size()
    if ws != 2:
        pytest.skip("Designed for torchrun WORLD_SIZE=2")
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    # Initialize FastVideo dist + model-parallel groups.
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=2)
    assert get_sp_group().world_size == 2

    yield

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    cleanup_dist_env_and_memory()


class ScaleModule(torch.nn.Module):
    def __init__(self, c: int, init: torch.Tensor):
        super().__init__()
        assert init.shape == (c,)
        self.scale = torch.nn.Parameter(init.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale.view(1, -1, 1, 1, 1)


def _broadcast_(t: torch.Tensor, src: int = 0) -> None:
    dist.broadcast(t, src=src)


def test_sp2_sharded_grad_matches_sp1_full_mse_grad(dist_setup_sp2):
    """
    Compare gradients from:
      (A) "SP disabled": full MSE mean((pred-target)^2)
      (B) "SP enabled": sp * local_sse / total_numel, where local_sse computed on
          a shard produced by shard_latents_across_sp(pred/target)

    Since training typically averages gradients across ranks, we simulate that by
    all-reducing and dividing by world_size for both gradients before comparing.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(2028)

    b, c, t, h, w = 2, 8, 1, 1, 9  # thw=9 -> requires padding for sp=2

    x = torch.empty((b, c, t, h, w), device=device, dtype=torch.float32)
    init = torch.empty((c,), device=device, dtype=torch.float32)
    if _rank() == 0:
        x.normal_()
        init.normal_()
    _broadcast_(x)
    _broadcast_(init)

    target = (0.25 * x).contiguous()

    # (A) "SP disabled" baseline: full MSE.
    m_full = ScaleModule(c=c, init=init).to(device)
    pred_full = m_full(x)
    loss_full = ((pred_full - target) ** 2).mean()
    loss_full.backward()
    grad_full = m_full.scale.grad.detach().clone()

    # (B) "SP enabled" sharded loss.
    m_sp = ScaleModule(c=c, init=init).to(device)
    pred = m_sp(x)
    sp = get_sp_group().world_size  # 2
    sharded_pred = shard_latents_across_sp(pred)
    sharded_target = shard_latents_across_sp(target)
    local_sse = ((sharded_pred - sharded_target) ** 2).sum()
    loss_sp = (sp * local_sse) / pred.numel()
    loss_sp.backward()
    grad_sp = m_sp.scale.grad.detach().clone()

    # Simulate gradient averaging across ranks (DDP/FSDP replicate behavior).
    dist.all_reduce(grad_full, op=dist.ReduceOp.SUM)
    dist.all_reduce(grad_sp, op=dist.ReduceOp.SUM)
    grad_full /= _world_size()
    grad_sp /= _world_size()

    torch.testing.assert_close(grad_sp, grad_full, rtol=1e-5, atol=1e-6)


