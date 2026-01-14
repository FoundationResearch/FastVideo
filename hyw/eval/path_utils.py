from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path) -> Path:
    """
    Find repo root by walking upward until we see pyproject.toml or .git.
    """
    p = start.resolve()
    while True:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
        if p == p.parent:
            raise RuntimeError("Could not find repo root (pyproject.toml/.git not found).")
        p = p.parent


def resolve_data_path(maybe_path: str, repo_root: Path) -> Path:
    """
    Resolve paths that may come from a different mount prefix (e.g. /mnt/fast-disks/...).

    Heuristic:
    - If path exists: use it.
    - Else if it contains '/FastVideo/': take suffix after that and join to current repo_root.
    """
    p = Path(maybe_path)
    if p.exists():
        return p

    marker = "/FastVideo/"
    s = str(maybe_path)
    if marker in s:
        suffix = s.split(marker, 1)[1]
        candidate = (repo_root / suffix).resolve()
        return candidate

    return p


