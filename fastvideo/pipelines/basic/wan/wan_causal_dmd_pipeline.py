# SPDX-License-Identifier: Apache-2.0
"""
Wan causal DMD pipeline implementation.

This module wires the causal DMD denoising stage into the modular pipeline.
"""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines import ComposedPipelineBase, LoRAPipeline
from fastvideo.pipelines.samplers.wan import get_wan_sampler_kind

# isort: off
from fastvideo.pipelines.stages import (ConditioningStage, DecodingStage,
                                        CausalDMDDenosingStage,
                                        CausalOdeDenoisingStage,
                                        InputValidationStage,
                                        LatentPreparationStage,
                                        TextEncodingStage,
                                        TimestepPreparationStage)
# isort: on

logger = init_logger(__name__)


class WanCausalDMDPipeline(LoRAPipeline, ComposedPipelineBase):

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler"
    ]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        """Set up pipeline stages with proper dependency injection."""
        sampler_kind = get_wan_sampler_kind(fastvideo_args)

        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())

        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))

        self.add_stage(stage_name="conditioning_stage",
                       stage=ConditioningStage())

        if sampler_kind == "ode":
            self.add_stage(
                stage_name="timestep_preparation_stage",
                stage=TimestepPreparationStage(
                    scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(
                           scheduler=self.get_module("scheduler"),
                           transformer=self.get_module("transformer", None)))

        if sampler_kind == "ode":
            denoising_stage = CausalOdeDenoisingStage(
                transformer=self.get_module("transformer"),
                transformer_2=self.get_module("transformer_2", None),
                scheduler=self.get_module("scheduler"),
                vae=self.get_module("vae"))
        else:
            denoising_stage = CausalDMDDenosingStage(
                transformer=self.get_module("transformer"),
                transformer_2=self.get_module("transformer_2", None),
                scheduler=self.get_module("scheduler"),
                vae=self.get_module("vae"))

        self.add_stage(stage_name="denoising_stage",
                       stage=denoising_stage)

        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae")))


EntryClass = WanCausalDMDPipeline
