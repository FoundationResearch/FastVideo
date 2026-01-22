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


@dataclass(frozen=True)
class _PipelineClassLocation:
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
            "Re-run with --like/--like-pipeline-class/--new-arch flags."
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
        return p

    def cmd(self, args: argparse.Namespace) -> None:
        repo_root = Path(args.repo_root).resolve()
        _ensure_repo_root(repo_root)

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
        if new_dir.exists():
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
            # Rename the base class declaration.
            base_text = re.sub(
                rf"^class\s+{re.escape(like_pipeline_class)}\s*\(",
                f"class {new_pipeline_class}(",
                base_text,
                flags=re.M,
            )
            # Rename EntryClass assignments.
            base_text = re.sub(
                rf"^EntryClass\s*=\s*{re.escape(like_pipeline_class)}\s*$",
                f"EntryClass = {new_pipeline_class}",
                base_text,
                flags=re.M,
            )
            pipeline_file.write_text(base_text, encoding="utf-8")

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


