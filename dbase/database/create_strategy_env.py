"""Copy strategy env config folders from a source env to a target env.

Comment density: orchestration

Mirrors DB ``create_test_environment`` for on-disk strategy configs under
``prod_strategies/<slug>/envs/<env>/`` (owned by the configs checkout via
``CONFIGS_DIR``). Lives in FinanceDatabase so environment create can seed
folders without importing TFP-Algo.

Core Functions:
    create_strategy_env: Copy one or more slug env trees.
    main: CLI entrypoint (``python -m dbase.database.create_strategy_env``).

Processing Flow:
    1. Resolve prod_strategies root (``CONFIGS_DIR`` required unless ``root`` passed).
    2. Discover slugs that have ``envs/<source>/`` (or use ``--slug``).
    3. Copy each source tree to ``envs/<target>/`` unless target exists
       without ``force``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Sequence


def _default_prod_strategies_root() -> Path:
    """Resolve the prod_strategies directory from ``CONFIGS_DIR``.

    Returns:
        Path: Absolute path to ``prod_strategies``.

    Raises:
        FileNotFoundError: If ``CONFIGS_DIR`` is unset or the path is missing.
    """
    configs_dir = os.environ.get("CONFIGS_DIR")
    if not configs_dir:
        raise FileNotFoundError(
            "CONFIGS_DIR is not set; cannot locate prod_strategies for env seeding. "
            "Set CONFIGS_DIR to the configs repo root, or pass root= explicitly."
        )
    return Path(configs_dir) / "prod_strategies"


def _list_slugs_with_source(root: Path, source_env: str) -> List[str]:
    """Return strategy slugs that already have ``envs/<source_env>/``.

    Args:
        root: ``prod_strategies`` directory.
        source_env: Source environment folder name.

    Returns:
        List[str]: Sorted slug names.
    """
    slugs: List[str] = []
    if not root.is_dir():
        return slugs
    for child in sorted(root.iterdir()):
        ## Skip shared configs package and non-strategy dirs
        if not child.is_dir() or child.name.startswith(".") or child.name == "configs":
            continue
        if (child / "envs" / source_env).is_dir():
            slugs.append(child.name)
    return slugs


def create_strategy_env(
    source_env: str,
    target_env: str,
    slugs: Optional[Sequence[str]] = None,
    force: bool = False,
    dry_run: bool = False,
    root: Optional[Path] = None,
) -> List[Path]:
    """Copy ``envs/<source_env>/`` to ``envs/<target_env>/`` for each slug.

    Args:
        source_env: Existing env folder name (e.g. ``live``).
        target_env: Destination env folder name (e.g. ``test``).
        slugs: Optional slug list; default = all slugs with the source env.
        force: Overwrite an existing target tree when True.
        dry_run: Print planned copies without writing.
        root: Optional ``prod_strategies`` root override.

    Returns:
        List[Path]: Target directories that were (or would be) created.

    Raises:
        FileNotFoundError: If root or a source env directory is missing.
        FileExistsError: If a target exists and ``force`` is False.
        ValueError: If source and target names are equal or empty.
    """
    if not source_env or not target_env:
        raise ValueError("source_env and target_env must be non-empty.")
    if source_env == target_env:
        raise ValueError("source_env and target_env must differ.")

    prod_root = root or _default_prod_strategies_root()
    if not prod_root.is_dir():
        raise FileNotFoundError(f"prod_strategies root not found: {prod_root}")

    selected = list(slugs) if slugs else _list_slugs_with_source(prod_root, source_env)
    if not selected:
        raise FileNotFoundError(
            f"No strategy slugs found with envs/{source_env}/ under {prod_root}"
        )

    created: List[Path] = []
    for slug in selected:
        source_dir = prod_root / slug / "envs" / source_env
        target_dir = prod_root / slug / "envs" / target_env
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing source env folder: {source_dir}")
        if target_dir.exists() and not force:
            raise FileExistsError(
                f"Target already exists (pass force=True / --force to overwrite): {target_dir}"
            )

        print(f"{'[dry-run] ' if dry_run else ''}{source_dir} -> {target_dir}")
        if not dry_run:
            if target_dir.exists():
                shutil.rmtree(target_dir)
            ## ignore __pycache__ if ever present under env trees
            shutil.copytree(
                source_dir,
                target_dir,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        created.append(target_dir)

    return created


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for create_strategy_env.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        argparse.Namespace: Parsed flags.
    """
    parser = argparse.ArgumentParser(
        description="Copy strategy env config folders from source to target."
    )
    parser.add_argument("--source-env", required=True, help="Source env folder name")
    parser.add_argument("--target-env", required=True, help="Target env folder name")
    parser.add_argument(
        "--slug",
        action="append",
        dest="slugs",
        default=None,
        help="Strategy slug (repeatable). Default: all slugs with source env.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing target env folders",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copies without writing",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint.

    Args:
        argv: Optional argument list.

    Returns:
        int: Process exit code (0 success, 1 error).
    """
    args = parse_args(argv)
    try:
        create_strategy_env(
            source_env=args.source_env,
            target_env=args.target_env,
            slugs=args.slugs,
            force=args.force,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
