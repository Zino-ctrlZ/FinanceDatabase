"""Unit tests for db_management sync-all fan-out."""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, call, patch

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_db_management_module():
    """Load db_management without importing dbase.database package __init__."""
    for name in (
        "pymysql",
        "mysql",
        "mysql.connector",
        "pandas",
        "sqlalchemy",
        "sqlalchemy.exc",
        "dotenv",
        "trade",
        "trade.helpers",
        "trade.helpers.helper",
        "trade.helpers.git",
    ):
        sys.modules.setdefault(name, MagicMock())
    sys.modules["trade.helpers.helper"].setup_logger = lambda *a, **k: MagicMock()

    db_utils = types.ModuleType("dbase.database.db_utils")

    class Database:
        PORTFOLIO_CONFIG = "portfolio_config"
        EXCLUDED_DATABASES = [
            "master_config",
            "information_schema",
            "mysql",
            "performance_schema",
            "sys",
        ]

    db_utils.Database = Database
    sys.modules["dbase"] = types.ModuleType("dbase")
    database_pkg = types.ModuleType("dbase.database")
    database_pkg.__path__ = [str(_REPO_ROOT / "dbase/database")]
    sys.modules["dbase.database"] = database_pkg
    sys.modules["dbase.database.db_utils"] = db_utils

    sql_helpers = types.ModuleType("dbase.database.SQLHelpers")
    sql_helpers.create_engine_short = MagicMock()
    sql_helpers.get_engine = MagicMock()
    sql_helpers.sql_host = "localhost"
    sql_helpers.sql_user = "user"
    sql_helpers.sql_pw = None
    sql_helpers.sql_port = "3306"
    sys.modules["dbase.database.SQLHelpers"] = sql_helpers

    path = _REPO_ROOT / "dbase/database/db_management.py"
    spec = importlib.util.spec_from_file_location(
        "dbase.database.db_management",
        path,
        submodule_search_locations=[str(_REPO_ROOT / "dbase/database")],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "dbase.database"
    sys.modules["dbase.database.db_management"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_db_management = _load_db_management_module()
build_cli_parser = _db_management.build_cli_parser
sync_all_environments_from_source = _db_management.sync_all_environments_from_source


def test_sync_all_skips_source_and_calls_pairwise_per_target():
    """Fan-out calls pairwise sync once per listed env except source."""
    listed = ["scratch", "long_bbands_v2", "test-mean-reversion"]
    pairwise = MagicMock(
        side_effect=lambda **kwargs: {
            "dry_run": not kwargs["apply"],
            "target": kwargs["target_environment"],
            "created_databases": {},
            "synced_tables": {},
        }
    )

    with (
        patch.object(_db_management, "list_environments", return_value=listed) as list_envs,
        patch.object(_db_management, "sync_environment_from_source", pairwise),
    ):
        result = sync_all_environments_from_source(
            source_environment="long_bbands_v2",
            apply=False,
        )

    list_envs.assert_called_once_with(exclude_prod=True)
    assert result["targets"] == ["scratch", "test-mean-reversion"]
    assert result["dry_run"] is True
    assert set(result["results"]) == {"scratch", "test-mean-reversion"}
    pairwise.assert_has_calls(
        [
            call(
                source_environment="long_bbands_v2",
                target_environment="scratch",
                branch_name=None,
                schema_only=True,
                copy_table_data=False,
                sync_databases=True,
                sync_tables=True,
                apply=False,
            ),
            call(
                source_environment="long_bbands_v2",
                target_environment="test-mean-reversion",
                branch_name=None,
                schema_only=True,
                copy_table_data=False,
                sync_databases=True,
                sync_tables=True,
                apply=False,
            ),
        ],
        any_order=False,
    )


def test_sync_all_apply_passes_apply_and_with_data():
    """Apply path forwards apply=True and copy_table_data to pairwise sync."""
    with (
        patch.object(_db_management, "list_environments", return_value=["scratch"]),
        patch.object(
            _db_management,
            "sync_environment_from_source",
            return_value={"dry_run": False},
        ) as pairwise,
    ):
        result = sync_all_environments_from_source(
            source_environment="prod",
            branch_name="feature-x",
            schema_only=False,
            copy_table_data=True,
            apply=True,
        )

    assert result["dry_run"] is False
    assert result["targets"] == ["scratch"]
    pairwise.assert_called_once_with(
        source_environment="prod",
        target_environment="scratch",
        branch_name="feature-x",
        schema_only=False,
        copy_table_data=True,
        sync_databases=True,
        sync_tables=True,
        apply=True,
    )


def test_sync_all_empty_targets_when_only_source_listed():
    """When list_environments returns only the source, targets is empty."""
    with (
        patch.object(_db_management, "list_environments", return_value=["scratch"]),
        patch.object(_db_management, "sync_environment_from_source") as pairwise,
    ):
        result = sync_all_environments_from_source(source_environment="scratch", apply=True)

    assert result["targets"] == []
    assert result["results"] == {}
    pairwise.assert_not_called()


def test_cli_sync_all_requires_source_env():
    """sync-all CLI requires --source-env."""
    parser = build_cli_parser()
    args = parser.parse_args(["sync-all", "--source-env", "prod", "--apply"])
    assert args.command == "sync-all"
    assert args.source_env == "prod"
    assert args.apply is True
