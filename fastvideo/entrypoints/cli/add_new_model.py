# SPDX-License-Identifier: Apache-2.0
"""
Developer utility: scaffold a brand-new FastVideo "model type" (all stubs).

This is complementary to `add-new-model-like`:
- `add-new-model`: create stubs for pipeline + configs + (optional) model stubs.
- `add-new-model-like`: fork from an existing implementation.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from fastvideo.entrypoints.cli.cli_types import CLISubcommand
from fastvideo.logger import init_logger
from fastvideo.utils import FlexibleArgumentParser

logger = init_logger(__name__)


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _to_pascal_case(s: str) -> str:
    parts = re.split(r"[^a-zA-Z0-9]+", s.strip())
    parts = [p for p in parts if p]
    if not parts:
        raise ValueError("new-arch must be non-empty")
    return "".join(p[:1].upper() + p[1:] for p in parts)


def _ensure_repo_root(repo_root: Path) -> None:
    if not (repo_root / "fastvideo").is_dir():
        raise ValueError(
            f"repo-root must contain a 'fastvideo/' directory, got: {repo_root}"
        )


def _insert_after_first_match(text: str, needle: str, insertion: str) -> str:
    if needle not in text:
        raise ValueError(f"Could not find marker: {needle}")
    return text.replace(needle, needle + "\n" + insertion, 1)


def _update_pipeline_registry_mapping(
    repo_root: Path,
    *,
    new_pipeline_class: str,
    new_arch: str,
) -> None:
    registry_path = repo_root / "fastvideo/pipelines/pipeline_registry.py"
    text = registry_path.read_text(encoding="utf-8")
    needle = "_PIPELINE_NAME_TO_ARCHITECTURE_NAME: dict[str, str] = {"
    entry = f'    "{new_pipeline_class}": "{new_arch}",\n'
    if entry in text:
        return
    text = _insert_after_first_match(text, needle, entry)
    registry_path.write_text(text, encoding="utf-8")


def _add_export_line(init_path: Path, import_line: str) -> None:
    text = init_path.read_text(encoding="utf-8")
    if import_line in text:
        return
    lines = text.splitlines(keepends=True)
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith("from "):
            insert_at = i + 1
    lines.insert(insert_at, import_line)
    init_path.write_text("".join(lines), encoding="utf-8")


class AddNewModelSubcommand(CLISubcommand):
    """
    `fastvideo add-new-model` - create a new architecture skeleton (all stubs).
    """

    def __init__(self) -> None:
        self.name = "add-new-model"
        super().__init__()

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        p = subparsers.add_parser(
            self.name,
            help="Create a brand-new model type (all stubs)",
        )
        p.add_argument(
            "--repo-root",
            type=str,
            default=".",
            help="Path to the FastVideo repo root (default: current directory)",
        )
        p.add_argument(
            "--new-arch",
            type=str,
            default="",
            help="New architecture name (snake-case recommended)",
        )
        p.add_argument(
            "--with-pipeline-config",
            action="store_true",
            help="Also create a PipelineConfig stub under fastvideo/configs/pipelines/",
        )
        p.add_argument(
            "--with-dit",
            action="store_true",
            help="Also create DiTConfig + DiT model stubs.",
        )
        p.add_argument(
            "--with-vae",
            action="store_true",
            help="Also create VAEConfig + VAE model stubs.",
        )
        p.add_argument(
            "--with-encoder",
            action="store_true",
            help="Also create TextEncoderConfig + TextEncoder model stubs.",
        )
        p.add_argument(
            "--with-scheduler",
            action="store_true",
            help="Also create a Scheduler stub under fastvideo/models/schedulers/.",
        )
        p.add_argument(
            "--export-configs",
            action="store_true",
            help="Also export new config classes in the corresponding __init__.py files.",
        )
        return p

    def cmd(self, args: argparse.Namespace) -> None:
        repo_root = Path(args.repo_root).resolve()
        _ensure_repo_root(repo_root)

        new_arch = str(args.new_arch).strip()
        if not new_arch:
            if not _is_interactive():
                raise ValueError("--new-arch is required in non-interactive mode.")
            new_arch = input("New architecture name: ").strip()
        if not new_arch:
            raise ValueError("--new-arch must be non-empty")

        pascal = _to_pascal_case(new_arch)
        pipeline_class = f"{pascal}Pipeline"
        pipeline_dir = repo_root / "fastvideo/pipelines/basic" / new_arch
        if pipeline_dir.exists():
            raise ValueError(f"Refusing to overwrite existing directory: {pipeline_dir}")

        # 1) Pipeline skeleton
        pipeline_dir.mkdir(parents=True, exist_ok=False)
        (pipeline_dir / "__init__.py").write_text(
            "# SPDX-License-Identifier: Apache-2.0\n"
            f'"""Pipelines for the {new_arch} architecture."""\n'
            f"from .{new_arch}_pipeline import {pipeline_class}\n\n"
            f"__all__ = [{pipeline_class!r}]\n",
            encoding="utf-8",
        )
        (pipeline_dir / f"{new_arch}_pipeline.py").write_text(
            "# SPDX-License-Identifier: Apache-2.0\n"
            "from __future__ import annotations\n"
            "\n"
            "from fastvideo.fastvideo_args import FastVideoArgs\n"
            "from fastvideo.pipelines import ComposedPipelineBase\n"
            "\n"
            "\n"
            f"class {pipeline_class}(ComposedPipelineBase):\n"
            f'    """Stub pipeline for {new_arch} (generated)."""\n'
            "\n"
            "    _required_config_modules = []\n"
            "\n"
            "    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):\n"
            "        raise NotImplementedError\n"
            "\n"
            "    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:\n"
            "        raise NotImplementedError\n"
            "\n"
            "\n"
            f"EntryClass = {pipeline_class}\n",
            encoding="utf-8",
        )
        _update_pipeline_registry_mapping(
            repo_root, new_pipeline_class=pipeline_class, new_arch=new_arch
        )

        # 2) Optional config/model stubs
        if args.with_dit:
            dit_cfg_path = repo_root / "fastvideo/configs/models/dits" / f"{new_arch}.py"
            dit_model_path = repo_root / "fastvideo/models/dits" / f"{new_arch}.py"
            dit_cfg_cls = f"{pascal}DiTConfig"
            dit_arch_cls = f"{pascal}ArchConfig"
            dit_model_cls = f"{pascal}Transformer3DModel"

            if dit_cfg_path.exists() or dit_model_path.exists():
                raise ValueError("Refusing to overwrite existing DiT files.")

            dit_cfg_path.write_text(
                "# SPDX-License-Identifier: Apache-2.0\n"
                "from __future__ import annotations\n"
                "\n"
                "from dataclasses import dataclass, field\n"
                "\n"
                "from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig\n"
                "\n"
                "\n"
                "@dataclass\n"
                f"class {dit_arch_cls}(DiTArchConfig):\n"
                "    # TODO: define real architecture fields\n"
                "    pass\n"
                "\n"
                "\n"
                "@dataclass\n"
                f"class {dit_cfg_cls}(DiTConfig):\n"
                f"    arch_config: DiTArchConfig = field(default_factory={dit_arch_cls})\n"
                f"    prefix: str = {new_arch!r}\n",
                encoding="utf-8",
            )
            dit_model_path.write_text(
                "# SPDX-License-Identifier: Apache-2.0\n"
                "from __future__ import annotations\n"
                "\n"
                "import torch\n"
                "\n"
                "from fastvideo.configs.models import DiTConfig\n"
                "from fastvideo.models.dits.base import BaseDiT\n"
                "\n"
                "\n"
                f"class {dit_model_cls}(BaseDiT):\n"
                "    _fsdp_shard_conditions = []\n"
                "    _compile_conditions = []\n"
                "    param_names_mapping = {}\n"
                "\n"
                "    def __init__(self, config: DiTConfig, hf_config: dict, **kwargs) -> None:\n"
                "        super().__init__(config=config, hf_config=hf_config, **kwargs)\n"
                "        self.hidden_size = 0\n"
                "        self.num_attention_heads = 0\n"
                "        self.num_channels_latents = 0\n"
                "\n"
                "    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states,\n"
                "                timestep: torch.LongTensor, encoder_hidden_states_image=None,\n"
                "                guidance=None, **kwargs) -> torch.Tensor:\n"
                "        raise NotImplementedError\n",
                encoding="utf-8",
            )
            if args.export_configs:
                _add_export_line(
                    repo_root / "fastvideo/configs/models/dits/__init__.py",
                    f"from fastvideo.configs.models.dits.{new_arch} import {dit_cfg_cls}\n",
                )

        if args.with_vae:
            vae_cfg_path = repo_root / "fastvideo/configs/models/vaes" / f"{new_arch}.py"
            vae_model_path = repo_root / "fastvideo/models/vaes" / f"{new_arch}.py"
            vae_cfg_cls = f"{pascal}VAEConfig"
            vae_arch_cls = f"{pascal}VAEArchConfig"
            vae_model_cls = f"{pascal}VAE"
            if vae_cfg_path.exists() or vae_model_path.exists():
                raise ValueError("Refusing to overwrite existing VAE files.")

            vae_cfg_path.write_text(
                "# SPDX-License-Identifier: Apache-2.0\n"
                "from __future__ import annotations\n"
                "\n"
                "from dataclasses import dataclass, field\n"
                "\n"
                "from fastvideo.configs.models.vaes.base import VAEArchConfig, VAEConfig\n"
                "\n"
                "\n"
                "@dataclass\n"
                f"class {vae_arch_cls}(VAEArchConfig):\n"
                "    # TODO: define real architecture fields\n"
                "    pass\n"
                "\n"
                "\n"
                "@dataclass\n"
                f"class {vae_cfg_cls}(VAEConfig):\n"
                f"    arch_config: VAEArchConfig = field(default_factory={vae_arch_cls})\n",
                encoding="utf-8",
            )
            vae_model_path.write_text(
                "# SPDX-License-Identifier: Apache-2.0\n"
                "from __future__ import annotations\n"
                "\n"
                "import torch\n"
                "\n"
                "from fastvideo.configs.models import VAEConfig\n"
                "from fastvideo.models.vaes.common import ParallelTiledVAE\n"
                "\n"
                "\n"
                f"class {vae_model_cls}(ParallelTiledVAE):\n"
                "    def _encode(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:\n"
                "        raise NotImplementedError\n"
                "\n"
                "    def _decode(self, z: torch.Tensor, *args, **kwargs) -> torch.Tensor:\n"
                "        raise NotImplementedError\n",
                encoding="utf-8",
            )
            if args.export_configs:
                _add_export_line(
                    repo_root / "fastvideo/configs/models/vaes/__init__.py",
                    f"from fastvideo.configs.models.vaes.{new_arch} import {vae_cfg_cls}\n",
                )

        if args.with_encoder:
            enc_cfg_path = repo_root / "fastvideo/configs/models/encoders" / f"{new_arch}.py"
            enc_model_path = repo_root / "fastvideo/models/encoders" / f"{new_arch}.py"
            enc_cfg_cls = f"{pascal}TextEncoderConfig"
            enc_arch_cls = f"{pascal}TextEncoderArchConfig"
            enc_model_cls = f"{pascal}TextEncoder"
            if enc_cfg_path.exists() or enc_model_path.exists():
                raise ValueError("Refusing to overwrite existing encoder files.")

            enc_cfg_path.write_text(
                "# SPDX-License-Identifier: Apache-2.0\n"
                "from __future__ import annotations\n"
                "\n"
                "from dataclasses import dataclass, field\n"
                "\n"
                "from fastvideo.configs.models.encoders.base import TextEncoderArchConfig, TextEncoderConfig\n"
                "\n"
                "\n"
                "@dataclass\n"
                f"class {enc_arch_cls}(TextEncoderArchConfig):\n"
                "    # TODO: define real architecture fields\n"
                "    pass\n"
                "\n"
                "\n"
                "@dataclass\n"
                f"class {enc_cfg_cls}(TextEncoderConfig):\n"
                f"    arch_config: TextEncoderArchConfig = field(default_factory={enc_arch_cls})\n"
                f"    prefix: str = {new_arch!r}\n",
                encoding="utf-8",
            )
            enc_model_path.write_text(
                "# SPDX-License-Identifier: Apache-2.0\n"
                "from __future__ import annotations\n"
                "\n"
                "import torch\n"
                "\n"
                "from fastvideo.configs.models.encoders import BaseEncoderOutput, TextEncoderConfig\n"
                "from fastvideo.models.encoders.base import TextEncoder\n"
                "\n"
                "\n"
                f"class {enc_model_cls}(TextEncoder):\n"
                "    def __init__(self, config: TextEncoderConfig) -> None:\n"
                "        super().__init__(config)\n"
                "\n"
                "    def forward(self, input_ids: torch.Tensor | None, position_ids=None,\n"
                "                attention_mask=None, inputs_embeds=None, output_hidden_states=None,\n"
                "                **kwargs) -> BaseEncoderOutput:\n"
                "        raise NotImplementedError\n",
                encoding="utf-8",
            )
            if args.export_configs:
                _add_export_line(
                    repo_root / "fastvideo/configs/models/encoders/__init__.py",
                    f"from fastvideo.configs.models.encoders.{new_arch} import {enc_cfg_cls}\n",
                )

        if args.with_scheduler:
            sched_path = repo_root / "fastvideo/models/schedulers" / f"{new_arch}.py"
            sched_cls = f"{pascal}Scheduler"
            if sched_path.exists():
                raise ValueError(f"Refusing to overwrite existing file: {sched_path}")
            sched_path.write_text(
                "# SPDX-License-Identifier: Apache-2.0\n"
                "from __future__ import annotations\n"
                "\n"
                "import torch\n"
                "\n"
                "from fastvideo.models.schedulers.base import BaseScheduler\n"
                "\n"
                "\n"
                f"class {sched_cls}(BaseScheduler):\n"
                "    # Minimal placeholders required by BaseScheduler\n"
                "    timesteps: torch.Tensor = torch.empty((0,), dtype=torch.int64)\n"
                "    order: int = 1\n"
                "    num_train_timesteps: int = 0\n"
                "\n"
                "    def set_shift(self, shift: float) -> None:\n"
                "        raise NotImplementedError\n"
                "\n"
                "    def set_timesteps(self, *args, **kwargs) -> None:\n"
                "        raise NotImplementedError\n"
                "\n"
                "    def scale_model_input(self, sample: torch.Tensor, timestep=None) -> torch.Tensor:\n"
                "        return sample\n",
                encoding="utf-8",
            )

        if args.with_pipeline_config:
            cfg_path = repo_root / "fastvideo/configs/pipelines" / f"{new_arch}.py"
            cfg_cls = f"{pascal}Config"
            if cfg_path.exists():
                raise ValueError(f"Refusing to overwrite existing file: {cfg_path}")

            # Wire to our stub configs if they were generated; otherwise fall back to base types.
            dit_cfg = f"{pascal}DiTConfig" if args.with_dit else "DiTConfig"
            vae_cfg = f"{pascal}VAEConfig" if args.with_vae else "VAEConfig"
            enc_cfg = f"{pascal}TextEncoderConfig" if args.with_encoder else "EncoderConfig"

            imports = [
                "from dataclasses import dataclass, field\n",
                "from fastvideo.configs.pipelines.base import PipelineConfig\n",
                "from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig\n",
            ]
            if args.with_dit:
                imports.append(
                    f"from fastvideo.configs.models.dits.{new_arch} import {dit_cfg}\n"
                )
            if args.with_vae:
                imports.append(
                    f"from fastvideo.configs.models.vaes.{new_arch} import {vae_cfg}\n"
                )
            if args.with_encoder:
                imports.append(
                    f"from fastvideo.configs.models.encoders.{new_arch} import {enc_cfg}\n"
                )

            cfg_path.write_text(
                "# SPDX-License-Identifier: Apache-2.0\n"
                "from __future__ import annotations\n"
                "\n"
                + "".join(imports)
                + "\n"
                "\n"
                "@dataclass\n"
                f"class {cfg_cls}(PipelineConfig):\n"
                f"    dit_config: DiTConfig = field(default_factory={dit_cfg})\n"
                f"    vae_config: VAEConfig = field(default_factory={vae_cfg})\n"
                f"    text_encoder_configs: tuple[EncoderConfig, ...] = field(default_factory=lambda: ({enc_cfg}(),))\n",
                encoding="utf-8",
            )
            if args.export_configs:
                _add_export_line(
                    repo_root / "fastvideo/configs/pipelines/__init__.py",
                    f"from fastvideo.configs.pipelines.{new_arch} import {cfg_cls}\n",
                )

        logger.info("Done. Created new stubs under arch '%s'.", new_arch)
        logger.info(
            "If you want a runnable fork of an existing model, use `fastvideo add-new-model-like`."
        )


def cmd_init() -> list[CLISubcommand]:
    return [AddNewModelSubcommand()]


