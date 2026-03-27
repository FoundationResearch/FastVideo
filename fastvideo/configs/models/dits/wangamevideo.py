# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models.dits.wanvideo import (
    WanVideoArchConfig,
    WanVideoConfig,
)


@dataclass
class WanGameVideoArchConfig(WanVideoArchConfig):
    """Wangame uses Wan defaults plus checkpoint-compat toggles."""

    image_cross_attn_type: str = "wangame"


@dataclass
class WanGameVideoConfig(WanVideoConfig):
    arch_config: WanGameVideoArchConfig = field(
        default_factory=WanGameVideoArchConfig
    )
    prefix: str = "WanGame"
