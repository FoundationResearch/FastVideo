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
from dataclasses import dataclass
from pathlib import Path

from fastvideo.entrypoints.cli.cli_types import CLISubcommand
from fastvideo.logger import init_logger
from fastvideo.utils import FlexibleArgumentParser

logger = init_logger(__name__)


_DEFAULT_LIKE_PIPELINE_CLASS: dict[str, str] = {
    "wan": "WanPipeline",
    "turbodiffusion": "TurboDiffusionPipeline",
    "stepvideo": "StepVideoPipeline",
    "hunyuan": "HunyuanVideoPipeline",
    "hunyuan15": "HunyuanVideo15Pipeline",
    "cosmos": "Cosmos2VideoToWorldPipeline",
    "matrixgame": "MatrixGamePipeline",
    "longcat": "LongCatPipeline",
    "ltx2": "LTX2Pipeline",
}


@dataclass(frozen=True)
class _PipelineClassLocation:
    module_relpath: str
    class_name: str


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


def _discover_pipeline_classes(arch_dir: Path) -> dict[str, _PipelineClassLocation]:
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
                    ),
                )
    return found


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
            "--repo-root",
            type=str,
            default=".",
            help="Path to the FastVideo repo root (default: current directory)",
        )
        p.add_argument(
            "--like",
            type=str,
            required=True,
            help=(
                "Existing pipeline architecture folder under "
                "fastvideo/pipelines/basic/ (e.g. wan, turbodiffusion, stepvideo)"
            ),
        )
        p.add_argument(
            "--new-arch",
            type=str,
            required=True,
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
        return p

    def cmd(self, args: argparse.Namespace) -> None:
        repo_root = Path(args.repo_root).resolve()
        _ensure_repo_root(repo_root)

        like_arch = str(args.like).strip()
        new_arch = str(args.new_arch).strip()
        if not like_arch or not new_arch:
            raise ValueError("--like and --new-arch must be non-empty")

        like_dir = repo_root / "fastvideo/pipelines/basic" / like_arch
        new_dir = repo_root / "fastvideo/pipelines/basic" / new_arch
        if not like_dir.is_dir():
            raise ValueError(f"--like arch not found: {like_dir}")
        if new_dir.exists():
            raise ValueError(
                f"Refusing to overwrite existing directory: {new_dir}"
            )

        available = _discover_pipeline_classes(like_dir)
        if not available:
            raise ValueError(
                f"No *Pipeline classes found under: {like_dir}. "
                "Expected at least one 'class ...Pipeline(...)' in a .py file."
            )

        like_pipeline_class = str(args.like_pipeline_class).strip()
        if not like_pipeline_class:
            like_pipeline_class = _DEFAULT_LIKE_PIPELINE_CLASS.get(like_arch, "")
        if not like_pipeline_class:
            raise ValueError(
                "Could not infer --like-pipeline-class. Available pipeline "
                f"classes in {like_arch}: {sorted(available.keys())}"
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
        (new_dir / f"{new_arch}_pipeline.py").write_text(
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
            f"    This was generated by `fastvideo {self.name}` and currently\n"
            f"    inherits all behavior from `{like_pipeline_class}`.\n"
            "    Override methods here as needed.\n"
            '    """\n'
            "\n"
            "    pass\n"
            "\n"
            "\n"
            f"EntryClass = {new_pipeline_class}\n",
            encoding="utf-8",
        )

        # 3) Update runtime registry mapping (pipeline name -> arch folder)
        _update_pipeline_registry_mapping(
            repo_root,
            new_pipeline_class=new_pipeline_class,
            new_arch=new_arch,
        )

        logger.info("Done. Created: %s", new_dir)
        logger.info(
            "Next: update your model_index.json _class_name to '%s' (or "
            "instantiate the pipeline explicitly).",
            new_pipeline_class,
        )


def cmd_init() -> list[CLISubcommand]:
    return [AddNewModelLikeSubcommand()]


