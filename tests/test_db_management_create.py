"""Unit tests for db_management create CLI (#15)."""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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
        "sqlalchemy",
        "dotenv",
        "trade",
        "trade.helpers",
        "trade.helpers.helper",
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
log_empty_bot_config_hint_after_create = _db_management.log_empty_bot_config_hint_after_create
create_test_environment = _db_management.create_test_environment


def test_create_parser_requires_source_env():
    parser = build_cli_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["create", "--env", "scratch", "--branch", "feature-x"])


def test_list_subcommand_registered():
    parser = build_cli_parser()
    args = parser.parse_args(["list"])
    assert args.command == "list"


def test_create_parser_accepts_source_env():
    parser = build_cli_parser()
    args = parser.parse_args(
        [
            "create",
            "--env",
            "scratch",
            "--branch",
            "feature-x",
            "--source-env",
            "long_bbands_v2",
            "--schema-only",
        ]
    )
    assert args.command == "create"
    assert args.env == "scratch"
    assert args.source_env == "long_bbands_v2"


def test_create_test_environment_requires_source_environment():
    with pytest.raises(ValueError, match="source_environment is required"):
        create_test_environment(
            environment="scratch",
            branch_name="feature-x",
            source_environment="",
        )


@patch.object(_db_management, "bot_config_table_row_counts")
@patch.object(_db_management, "db_management_logger")
def test_hint_logged_when_bot_tables_empty(mock_logger, mock_counts):
    mock_counts.return_value = {
        "discord_channels": 0,
        "bot_identity": 0,
        "bot_clients": 0,
    }
    log_empty_bot_config_hint_after_create(
        target_environment="scratch",
        source_environment="long_bbands_v2",
        created={"portfolio_config": "portfolio_config_scratch"},
        schema_only=True,
    )
    mock_counts.assert_called_once_with("portfolio_config_scratch")
    mock_logger.info.assert_called_once()
    message = mock_logger.info.call_args[0][0]
    assert "seed_bot_config" in message
    assert "--env scratch" in message
    assert "--from-source-env long_bbands_v2" in message
    assert "verify_bot_config" in message


@patch.object(_db_management, "bot_config_table_row_counts")
@patch.object(_db_management, "db_management_logger")
def test_hint_skipped_when_rows_present(mock_logger, mock_counts):
    mock_counts.return_value = {
        "discord_channels": 1,
        "bot_identity": 0,
        "bot_clients": 0,
    }
    log_empty_bot_config_hint_after_create(
        target_environment="scratch",
        source_environment="long_bbands_v2",
        created={"portfolio_config": "portfolio_config_scratch"},
        schema_only=True,
    )
    mock_logger.info.assert_not_called()


@patch.object(_db_management, "bot_config_table_row_counts")
@patch.object(_db_management, "db_management_logger")
def test_hint_skipped_when_not_schema_only(mock_logger, mock_counts):
    log_empty_bot_config_hint_after_create(
        target_environment="scratch",
        source_environment="long_bbands_v2",
        created={"portfolio_config": "portfolio_config_scratch"},
        schema_only=False,
    )
    mock_counts.assert_not_called()
    mock_logger.info.assert_not_called()


@patch.object(_db_management, "bot_config_table_row_counts")
@patch.object(_db_management, "db_management_logger")
def test_hint_skipped_when_portfolio_config_not_created(mock_logger, mock_counts):
    log_empty_bot_config_hint_after_create(
        target_environment="scratch",
        source_environment="long_bbands_v2",
        created={"portfolio_data": "portfolio_data_scratch"},
        schema_only=True,
    )
    mock_counts.assert_not_called()
    mock_logger.info.assert_not_called()


@patch.object(_db_management, "bot_config_table_row_counts", side_effect=RuntimeError("no table"))
@patch.object(_db_management, "db_management_logger")
def test_hint_failure_does_not_raise(mock_logger, _mock_counts):
    log_empty_bot_config_hint_after_create(
        target_environment="scratch",
        source_environment="long_bbands_v2",
        created={"portfolio_config": "portfolio_config_scratch"},
        schema_only=True,
    )
    mock_logger.info.assert_not_called()
    mock_logger.debug.assert_called_once()
