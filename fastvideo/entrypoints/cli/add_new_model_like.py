# SPDX-License-Identifier: Apache-2.0
"""
Developer utility: scaffold a new FastVideo pipeline "like" an existing one.

This is inspired by `transformers-cli add-new-model-like`, but tailored to
FastVideo's pipeline + registry structure.

Design goals:
- Create runnable code immediately (prefer inheritance over copy/paste).
- Make the new pipeline discoverable by FastVideo runtime registry.
- Be safe by default (fail fast if files already exist).
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from fastvideo.entrypoints.cli.cli_types import CLISubcommand
from fastvideo.logger import init_logger
from fastvideo.utils import FlexibleArgumentParser

logger = init_logger(__name__)

_COMPONENT_CHOICES = (
    # Pipeline layer
    "pipeline",  # fastvideo/pipelines/basic/<new_arch>/*
    "pipeline_registry",  # fastvideo/pipelines/pipeline_registry.py mapping
    # Config layer
    "pipeline_config",  # fastvideo/configs/pipelines/<new_arch>.py
    "pipelines_init",  # fastvideo/configs/pipelines/__init__.py export
    # Model layer (stubs)
    "dit_config",  # fastvideo/configs/models/dits/<new_arch>.py (stub)
    "dits_init",  # fastvideo/configs/models/dits/__init__.py export
    "dit_model",  # fastvideo/models/dits/<new_arch>.py (stub)
)


@dataclass(frozen=True)
class _PipelineClassLocation:
    module_relpath: str
    class_name: str
    file_abs: Path


@dataclass(frozen=True)
class _ConfigClassLocation:
    module_relpath: str
    class_name: str
    file_abs: Path


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


def _discover_pipeline_classes_in_arch_dir(
    arch_dir: Path,
) -> dict[str, _PipelineClassLocation]:
    """
    Return mapping {PipelineClassName -> location} for a given
    fastvideo/pipelines/basic/<arch> directory.
    """
    if not arch_dir.is_dir():
        raise ValueError(f"Pipeline arch directory not found: {arch_dir}")

    found: dict[str, _PipelineClassLocation] = {}
    for py in sorted(arch_dir.glob("*.py")):
        text = py.read_text(encoding="utf-8")
        for m in re.finditer(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", text, re.M):
            cls = m.group(1)
            if cls.endswith("Pipeline"):
                found.setdefault(
                    cls,
                    _PipelineClassLocation(
                        module_relpath=str(py.relative_to(arch_dir.parent.parent)),
                        class_name=cls,
                        file_abs=py,
                    ),
                )
    return found


def _discover_all_basic_pipelines(
    repo_root: Path,
) -> dict[str, dict[str, _PipelineClassLocation]]:
    """
    Return {arch -> {PipelineClassName -> location}} for fastvideo/pipelines/basic/*.
    """
    base_dir = repo_root / "fastvideo/pipelines/basic"
    if not base_dir.is_dir():
        raise ValueError(f"basic pipelines directory not found: {base_dir}")

    out: dict[str, dict[str, _PipelineClassLocation]] = {}
    for arch_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        arch = arch_dir.name
        out[arch] = _discover_pipeline_classes_in_arch_dir(arch_dir)
    return out


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _prompt_choice(title: str, options: list[str]) -> str:
    if not _is_interactive():
        raise RuntimeError(
            "Interactive selection requires a TTY. "
            "Re-run with --like/--like-pipeline-class/--new-arch (and/or --components) flags."
        )
    if not options:
        raise ValueError("No options available for selection.")
    print(title)
    for i, opt in enumerate(options, start=1):
        print(f"  [{i}] {opt}")
    while True:
        raw = input("Select an option: ").strip()
        try:
            idx = int(raw)
        except ValueError:
            print("Please enter a number.")
            continue
        if 1 <= idx <= len(options):
            return options[idx - 1]
        print(f"Please enter a number between 1 and {len(options)}.")


def _prompt_yes_no(question: str, *, default: bool) -> bool:
    if not _is_interactive():
        return default
    suffix = " [Y/n] " if default else " [y/N] "
    while True:
        raw = input(question + suffix).strip().lower()
        if raw == "":
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please answer y/n.")


def _prompt_components() -> set[str]:
    """
    Interactive selection of which parts to generate.
    """
    selected: set[str] = {"pipeline", "pipeline_registry"}

    if _prompt_yes_no(
            "Also generate a PipelineConfig (fastvideo/configs/pipelines/*.py)?",
            default=False):
        selected.add("pipeline_config")
        if _prompt_yes_no(
                "Export it from fastvideo/configs/pipelines/__init__.py?",
                default=True):
            selected.add("pipelines_init")

    if _prompt_yes_no(
            "Also generate a DiTConfig stub (fastvideo/configs/models/dits/*.py)?",
            default=False):
        selected.add("dit_config")
        if _prompt_yes_no(
                "Export it from fastvideo/configs/models/dits/__init__.py?",
                default=True):
            selected.add("dits_init")

    if _prompt_yes_no("Also generate a DiT model stub (fastvideo/models/dits/*.py)?",
                      default=False):
        selected.add("dit_model")

    return selected


def _update_pipeline_registry_mapping(
    repo_root: Path,
    *,
    new_pipeline_class: str,
    new_arch: str,
) -> None:
    registry_path = repo_root / "fastvideo/pipelines/pipeline_registry.py"
    if not registry_path.exists():
        raise ValueError(f"pipeline registry not found: {registry_path}")

    text = registry_path.read_text(encoding="utf-8")
    # Keep this intentionally simple/explicit: insert right after the dict header.
    # We avoid clever parsing to keep the tool robust.
    needle = "_PIPELINE_NAME_TO_ARCHITECTURE_NAME: dict[str, str] = {"
    if needle not in text:
        raise ValueError(
            "Could not find _PIPELINE_NAME_TO_ARCHITECTURE_NAME in "
            f"{registry_path}"
        )
    entry = f'    "{new_pipeline_class}": "{new_arch}",\n'
    if entry in text:
        return
    text = text.replace(needle, needle + "\n" + entry)
    registry_path.write_text(text, encoding="utf-8")


def _discover_pipeline_config_classes(
        repo_root: Path) -> dict[str, _ConfigClassLocation]:
    """
    Discover classes like `class XxxConfig(PipelineConfig):` in
    fastvideo/configs/pipelines/*.py
    """
    cfg_dir = repo_root / "fastvideo/configs/pipelines"
    if not cfg_dir.is_dir():
        raise ValueError(f"pipelines config dir not found: {cfg_dir}")

    found: dict[str, _ConfigClassLocation] = {}
    for py in sorted(cfg_dir.glob("*.py")):
        text = py.read_text(encoding="utf-8")
        for m in re.finditer(
                r"^class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*PipelineConfig\s*\)\s*:",
                text,
                re.M,
        ):
            cls = m.group(1)
            found.setdefault(
                cls,
                _ConfigClassLocation(
                    module_relpath=str(py.relative_to(cfg_dir.parent.parent)),
                    class_name=cls,
                    file_abs=py,
                ),
            )
    return found


def _add_export_to_pipelines_init(repo_root: Path, *, class_name: str,
                                 module_name: str) -> None:
    init_path = repo_root / "fastvideo/configs/pipelines/__init__.py"
    text = init_path.read_text(encoding="utf-8")
    import_line = (
        f"from fastvideo.configs.pipelines.{module_name} import {class_name}\n")

    if import_line not in text:
        lines = text.splitlines(keepends=True)
        insert_at = 0
        for i, line in enumerate(lines):
            if line.startswith("from "):
                insert_at = i + 1
        lines.insert(insert_at, import_line)
        text = "".join(lines)

    if "__all__" in text and f'"{class_name}"' not in text:
        text = re.sub(r"(__all__\s*=\s*\[)",
                      r"\1\n    " + f'"{class_name}", ',
                      text,
                      count=1)

    init_path.write_text(text, encoding="utf-8")


def _add_export_to_dits_init(repo_root: Path, *, class_name: str,
                             module_name: str) -> None:
    init_path = repo_root / "fastvideo/configs/models/dits/__init__.py"
    text = init_path.read_text(encoding="utf-8")
    import_line = (
        f"from fastvideo.configs.models.dits.{module_name} import {class_name}\n"
    )

    if import_line not in text:
        lines = text.splitlines(keepends=True)
        insert_at = 0
        for i, line in enumerate(lines):
            if line.startswith("from "):
                insert_at = i + 1
        lines.insert(insert_at, import_line)
        text = "".join(lines)

    if "__all__" in text and f'"{class_name}"' not in text:
        text = re.sub(r"(__all__\s*=\s*\[)",
                      r"\1\n    " + f'"{class_name}", ',
                      text,
                      count=1)

    init_path.write_text(text, encoding="utf-8")


def _write_pipeline_config(
    repo_root: Path,
    *,
    new_arch: str,
    new_config_class: str,
    base_config_loc: _ConfigClassLocation,
) -> None:
    cfg_dir = repo_root / "fastvideo/configs/pipelines"
    out = cfg_dir / f"{new_arch}.py"
    if out.exists():
        raise ValueError(f"Refusing to overwrite existing file: {out}")

    module_path = base_config_loc.module_relpath.replace("/", ".")
    if module_path.endswith(".py"):
        module_path = module_path[:-3]

    base_cls = base_config_loc.class_name
    out.write_text(
        "# SPDX-License-Identifier: Apache-2.0\n"
        "from __future__ import annotations\n"
        "\n"
        "from dataclasses import dataclass\n"
        "\n"
        f"from {module_path} import {base_cls}\n"
        "\n"
        "\n"
        "@dataclass\n"
        f"class {new_config_class}({base_cls}):\n"
        '    """\n'
        f"    Scaffolded PipelineConfig for '{new_arch}'.\n"
        "\n"
        "    Generated by `fastvideo add-new-model-like`.\n"
        f"    Base: `{base_cls}`.\n"
        '    """\n'
        "\n"
        "    pass\n",
        encoding="utf-8",
    )


def _write_dit_config_stub(
    repo_root: Path,
    *,
    new_arch: str,
    new_config_class: str,
) -> None:
    cfg_dir = repo_root / "fastvideo/configs/models/dits"
    out = cfg_dir / f"{new_arch}.py"
    if out.exists():
        raise ValueError(f"Refusing to overwrite existing file: {out}")

    arch_cls = f"{_to_pascal_case(new_arch)}ArchConfig"
    out.write_text(
        "# SPDX-License-Identifier: Apache-2.0\n"
        "from __future__ import annotations\n"
        "\n"
        "from dataclasses import dataclass, field\n"
        "\n"
        "from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig\n"
        "\n"
        "\n"
        "@dataclass\n"
        f"class {arch_cls}(DiTArchConfig):\n"
        '    """Architecture fields for the new DiT (stub)."""\n'
        "\n"
        "    # TODO: fill real architecture fields\n"
        "    pass\n"
        "\n"
        "\n"
        "@dataclass\n"
        f"class {new_config_class}(DiTConfig):\n"
        f"    arch_config: DiTArchConfig = field(default_factory={arch_cls})\n"
        "\n"
        f"    prefix: str = {new_arch!r}\n",
        encoding="utf-8",
    )


def _write_dit_model_stub(
    repo_root: Path,
    *,
    new_arch: str,
    new_model_class: str,
) -> None:
    out_dir = repo_root / "fastvideo/models/dits"
    out = out_dir / f"{new_arch}.py"
    if out.exists():
        raise ValueError(f"Refusing to overwrite existing file: {out}")

    out.write_text(
        "# SPDX-License-Identifier: Apache-2.0\n"
        "from __future__ import annotations\n"
        "\n"
        "import torch\n"
        "\n"
        "from fastvideo.configs.models import DiTConfig\n"
        "from fastvideo.models.dits.base import BaseDiT\n"
        "\n"
        "\n"
        f"class {new_model_class}(BaseDiT):\n"
        '    """\n'
        f"    Stub DiT model for '{new_arch}'.\n"
        "\n"
        "    NOTE: This is NOT a functional implementation. It exists so imports\n"
        "    are wired and you have a starting point.\n"
        '    """\n'
        "\n"
        "    _fsdp_shard_conditions = []\n"
        "    _compile_conditions = []\n"
        "    param_names_mapping = {}\n"
        "\n"
        "    def __init__(self, config: DiTConfig, hf_config: dict, **kwargs) -> None:\n"
        "        super().__init__(config=config, hf_config=hf_config, **kwargs)\n"
        "        # TODO: set required instance attrs\n"
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


class AddNewModelLikeSubcommand(CLISubcommand):
    """
    `fastvideo add-new-model-like` - scaffold a new pipeline architecture.

    Example:
        fastvideo add-new-model-like --like wan --new-arch myarch
    """

    def __init__(self) -> None:
        self.name = "add-new-model-like"
        super().__init__()

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        p = subparsers.add_parser(
            self.name,
            help="Scaffold a new pipeline architecture like an existing one",
        )
        p.add_argument(
            "--list-like",
            action="store_true",
            help="List available --like architectures and pipeline classes, then exit.",
        )
        p.add_argument(
            "--repo-root",
            type=str,
            default=".",
            help="Path to the FastVideo repo root (default: current directory)",
        )
        p.add_argument(
            "--components",
            action="append",
            default=[],
            choices=_COMPONENT_CHOICES,
            help=(
                "Which parts to generate. Repeatable. If omitted in a TTY, "
                "FastVideo will launch an interactive wizard. If omitted in a "
                "non-interactive context, defaults to: pipeline + pipeline_registry."
            ),
        )
        p.add_argument(
            "--like",
            type=str,
            default="",
            help=(
                "Existing pipeline architecture folder under "
                "fastvideo/pipelines/basic/ (e.g. wan, turbodiffusion, stepvideo)"
            ),
        )
        p.add_argument(
            "--new-arch",
            type=str,
            default="",
            help=(
                "New pipeline architecture folder name under "
                "fastvideo/pipelines/basic/ (snake-case recommended)"
            ),
        )
        p.add_argument(
            "--like-pipeline-class",
            type=str,
            default="",
            help=(
                "Pipeline class name to inherit from. If omitted, FastVideo will "
                "use a sensible default per architecture."
            ),
        )
        p.add_argument(
            "--new-pipeline-class",
            type=str,
            default="",
            help=(
                "New pipeline class name. If omitted, uses "
                "<NewArchPascalCase>Pipeline."
            ),
        )
        p.add_argument(
            "--template",
            type=str,
            choices=["inherit", "copy"],
            default="inherit",
            help=(
                "How to scaffold the new pipeline implementation. "
                "'inherit' creates a subclass of the base pipeline (reuse stages). "
                "'copy' copies the base pipeline module text and renames the class."
            ),
        )
        p.add_argument(
            "--new-stages",
            action="store_true",
            help=(
                "Generate stubs for initialize_pipeline/create_pipeline_stages "
                "instead of reusing the base pipeline's stage wiring. "
                "Best with --template=inherit."
            ),
        )
        p.add_argument(
            "--like-pipeline-config-class",
            type=str,
            default="",
            help=(
                "Base PipelineConfig class to inherit from when generating "
                "--components pipeline_config."
            ),
        )
        p.add_argument(
            "--new-pipeline-config-class",
            type=str,
            default="",
            help=(
                "New PipelineConfig class name. Default: "
                "<NewArchPascalCase>Config."
            ),
        )
        p.add_argument(
            "--new-dit-config-class",
            type=str,
            default="",
            help="New DiTConfig class name (for --components dit_config).",
        )
        p.add_argument(
            "--new-dit-model-class",
            type=str,
            default="",
            help="New DiT model class name (for --components dit_model).",
        )
        return p

    def cmd(self, args: argparse.Namespace) -> None:
        repo_root = Path(args.repo_root).resolve()
        _ensure_repo_root(repo_root)

        components = set(args.components or [])
        if not components:
            if _is_interactive():
                components = _prompt_components()
            else:
                components = {"pipeline", "pipeline_registry"}

        all_pipelines = _discover_all_basic_pipelines(repo_root)

        if args.list_like:
            for arch, cls_map in sorted(all_pipelines.items()):
                if not cls_map:
                    continue
                print(f"{arch}:")
                for cls in sorted(cls_map.keys()):
                    print(f"  - {cls}")
            return

        like_arch = str(args.like).strip()
        new_arch = str(args.new_arch).strip()

        # Interactive UX (transformers-cli style): if user doesn't provide inputs,
        # prompt them.
        if not like_arch:
            like_arch = _prompt_choice(
                "Pick a base pipeline architecture to copy from:",
                [a for a, m in sorted(all_pipelines.items()) if m],
            )
        if not new_arch:
            if not _is_interactive():
                raise ValueError("--new-arch is required in non-interactive mode.")
            new_arch = input("New architecture name (folder under fastvideo/pipelines/basic/): ").strip()
            if not new_arch:
                raise ValueError("--new-arch must be non-empty")

        like_dir = repo_root / "fastvideo/pipelines/basic" / like_arch
        new_dir = repo_root / "fastvideo/pipelines/basic" / new_arch
        if not like_dir.is_dir():
            raise ValueError(f"--like arch not found: {like_dir}")
        if "pipeline" in components and new_dir.exists():
            raise ValueError(f"Refusing to overwrite existing directory: {new_dir}")

        available = all_pipelines.get(like_arch, {})
        if not available:
            raise ValueError(
                f"No *Pipeline classes found under: {like_dir}. "
                "Expected at least one 'class ...Pipeline(...)' in a .py file."
            )

        like_pipeline_class = str(args.like_pipeline_class).strip()
        if not like_pipeline_class:
            # If only one pipeline class exists for this arch, default to it.
            if len(available) == 1:
                like_pipeline_class = next(iter(available.keys()))
            else:
                like_pipeline_class = _prompt_choice(
                    "Pick a base pipeline class:",
                    sorted(available.keys()),
                )
        if like_pipeline_class not in available:
            raise ValueError(
                f"--like-pipeline-class={like_pipeline_class} not found. "
                f"Available: {sorted(available.keys())}"
            )

        new_pipeline_class = str(args.new_pipeline_class).strip()
        if not new_pipeline_class:
            new_pipeline_class = f"{_to_pascal_case(new_arch)}Pipeline"

        loc = available[like_pipeline_class]
        # loc.module_relpath looks like "fastvideo/pipelines/basic/wan/wan_pipeline.py"
        module_path = loc.module_relpath.replace("/", ".")
        if module_path.endswith(".py"):
            module_path = module_path[:-3]

        logger.info(
            "Scaffolding new arch '%s' like '%s' (base=%s from %s)",
            new_arch,
            like_arch,
            like_pipeline_class,
            module_path,
        )

        if "pipeline" in components:
            new_dir.mkdir(parents=True, exist_ok=False)

            # 1) __init__.py
            (new_dir / "__init__.py").write_text(
                "# SPDX-License-Identifier: Apache-2.0\n"
                f'"""Pipelines for the {new_arch} architecture."""\n'
                f"from .{new_arch}_pipeline import {new_pipeline_class}\n\n"
                f"__all__ = [{new_pipeline_class!r}]\n",
                encoding="utf-8",
            )

            # 2) pipeline module
            pipeline_file = new_dir / f"{new_arch}_pipeline.py"
            if args.template == "inherit":
                body = (
                    "# SPDX-License-Identifier: Apache-2.0\n"
                    "\n"
                    "from __future__ import annotations\n"
                    "\n"
                    f"from {module_path} import {like_pipeline_class}\n"
                    "\n"
                    "\n"
                    f"class {new_pipeline_class}({like_pipeline_class}):\n"
                    '    """\n'
                    f"    Scaffolded pipeline for {new_arch}.\n"
                    "\n"
                    f"    Generated by `fastvideo {self.name}`.\n"
                    f"    Base: `{like_pipeline_class}`.\n"
                    '    """\n'
                )
                if args.new_stages:
                    body += (
                        "\n"
                        "    # NOTE: you opted into --new-stages, so this class does not\n"
                        "    # reuse the base pipeline's stage wiring. Fill these in.\n"
                        "    def initialize_pipeline(self, fastvideo_args):\n"
                        "        raise NotImplementedError\n"
                        "\n"
                        "    def create_pipeline_stages(self, fastvideo_args):\n"
                        "        raise NotImplementedError\n"
                        "\n"
                    )
                else:
                    body += "\n    pass\n\n"
                body += f"\nEntryClass = {new_pipeline_class}\n"
                pipeline_file.write_text(body, encoding="utf-8")
            else:
                # Copy the base module text and do minimal renames.
                base_text = loc.file_abs.read_text(encoding="utf-8")
                base_text = (
                    "# SPDX-License-Identifier: Apache-2.0\n"
                    "# NOTE: generated by `fastvideo add-new-model-like --template=copy`.\n"
                    + "\n"
                    + base_text
                )
                base_text = re.sub(
                    rf"^class\s+{re.escape(like_pipeline_class)}\s*\(",
                    f"class {new_pipeline_class}(",
                    base_text,
                    flags=re.M,
                )
                base_text = re.sub(
                    rf"^EntryClass\s*=\s*{re.escape(like_pipeline_class)}\s*$",
                    f"EntryClass = {new_pipeline_class}",
                    base_text,
                    flags=re.M,
                )
                pipeline_file.write_text(base_text, encoding="utf-8")

        if "pipeline_registry" in components:
            _update_pipeline_registry_mapping(
                repo_root,
                new_pipeline_class=new_pipeline_class,
                new_arch=new_arch,
            )

        if "pipeline_config" in components:
            cfgs = _discover_pipeline_config_classes(repo_root)
            like_cfg = str(args.like_pipeline_config_class).strip()
            if not like_cfg:
                like_cfg = _prompt_choice("Pick a base PipelineConfig class:",
                                          sorted(cfgs.keys()))
            if like_cfg not in cfgs:
                raise ValueError(
                    f"--like-pipeline-config-class={like_cfg} not found. "
                    f"Available: {sorted(cfgs.keys())}"
                )
            new_cfg_cls = str(args.new_pipeline_config_class).strip()
            if not new_cfg_cls:
                new_cfg_cls = f"{_to_pascal_case(new_arch)}Config"
            _write_pipeline_config(repo_root,
                                   new_arch=new_arch,
                                   new_config_class=new_cfg_cls,
                                   base_config_loc=cfgs[like_cfg])
            if "pipelines_init" in components:
                _add_export_to_pipelines_init(repo_root,
                                              class_name=new_cfg_cls,
                                              module_name=new_arch)

        if "dit_config" in components:
            new_dit_cfg_cls = str(args.new_dit_config_class).strip()
            if not new_dit_cfg_cls:
                new_dit_cfg_cls = f"{_to_pascal_case(new_arch)}DiTConfig"
            _write_dit_config_stub(repo_root,
                                   new_arch=new_arch,
                                   new_config_class=new_dit_cfg_cls)
            if "dits_init" in components:
                _add_export_to_dits_init(repo_root,
                                         class_name=new_dit_cfg_cls,
                                         module_name=new_arch)

        if "dit_model" in components:
            new_dit_model_cls = str(args.new_dit_model_class).strip()
            if not new_dit_model_cls:
                new_dit_model_cls = f"{_to_pascal_case(new_arch)}Transformer3DModel"
            _write_dit_model_stub(repo_root,
                                  new_arch=new_arch,
                                  new_model_class=new_dit_model_cls)

        logger.info("Done.")
        logger.info(
            "Next: update your model_index.json _class_name to '%s' (or "
            "instantiate the pipeline explicitly).",
            new_pipeline_class,
        )


def cmd_init() -> list[CLISubcommand]:
    return [AddNewModelLikeSubcommand()]


