"""Unit tests for strategy env folder seeding."""

from __future__ import annotations

from pathlib import Path

import pytest

from dbase.database.create_strategy_env import create_strategy_env


def _seed_source(root: Path, slug: str, env: str) -> Path:
    """Create a minimal source env tree under ``root``."""

    env_dir = root / slug / "envs" / env
    env_dir.mkdir(parents=True)
    (env_dir / "params.json").write_text("{}", encoding="utf-8")
    return env_dir


def test_create_strategy_env_copies_tree(tmp_path: Path) -> None:
    """Source env folder is copied to the target env name."""

    root = tmp_path / "prod_strategies"
    _seed_source(root, "demo_slug", "live")

    created = create_strategy_env(
        source_env="live",
        target_env="test",
        root=root,
    )
    assert len(created) == 1
    assert created[0] == root / "demo_slug" / "envs" / "test"
    assert (created[0] / "params.json").is_file()


def test_create_strategy_env_refuses_existing_without_force(tmp_path: Path) -> None:
    """Existing target raises unless force=True."""

    root = tmp_path / "prod_strategies"
    _seed_source(root, "demo_slug", "live")
    create_strategy_env(source_env="live", target_env="test", root=root)

    with pytest.raises(FileExistsError):
        create_strategy_env(source_env="live", target_env="test", root=root)

    create_strategy_env(source_env="live", target_env="test", root=root, force=True)


def test_create_strategy_env_missing_source(tmp_path: Path) -> None:
    """Missing source env raises FileNotFoundError."""

    root = tmp_path / "prod_strategies"
    root.mkdir()
    (root / "demo_slug").mkdir()

    with pytest.raises(FileNotFoundError):
        create_strategy_env(source_env="live", target_env="test", root=root)
