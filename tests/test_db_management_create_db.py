"""Unit tests for db_management create-db (single empty database)."""

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
resolve_physical_database_name = _db_management.resolve_physical_database_name
create_database_for_environment = _db_management.create_database_for_environment


def test_resolve_physical_database_name_prod():
    assert resolve_physical_database_name("portfolio_data", "prod") == "portfolio_data"


def test_resolve_physical_database_name_non_prod():
    assert (
        resolve_physical_database_name("portfolio_config", "scratch")
        == "portfolio_config_scratch"
    )


def test_create_db_parser_requires_env_and_name():
    parser = build_cli_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["create-db", "--env", "scratch"])
    with pytest.raises(SystemExit):
        parser.parse_args(["create-db", "--name", "portfolio_data"])
    args = parser.parse_args(["create-db", "--env", "scratch", "--name", "portfolio_data"])
    assert args.command == "create-db"
    assert args.env == "scratch"
    assert args.name == "portfolio_data"


def test_excluded_base_name_raises():
    with pytest.raises(ValueError, match="protected"):
        create_database_for_environment(
            base_name="master_config",
            environment="scratch",
            branch_name="feature-x",
        )


@patch.object(_db_management, "register_database")
@patch.object(_db_management, "check_database_conflict")
@patch.object(_db_management, "mysql_database_exists", return_value=False)
@patch.object(_db_management, "create_engine_short")
def test_dry_run_skips_create_and_register(
    mock_create_engine,
    _mock_exists,
    mock_conflict,
    mock_register,
):
    name = create_database_for_environment(
        base_name="portfolio_data",
        environment="scratch",
        branch_name="feature-x",
        dry_run=True,
    )
    assert name == "portfolio_data_scratch"
    mock_create_engine.assert_not_called()
    mock_conflict.assert_not_called()
    mock_register.assert_not_called()


@patch.object(_db_management, "register_database")
@patch.object(_db_management, "check_database_conflict")
@patch.object(_db_management, "mysql_database_exists", return_value=True)
def test_mysql_exists_raises(_mock_exists, mock_conflict, mock_register):
    with pytest.raises(ValueError, match="already exists"):
        create_database_for_environment(
            base_name="portfolio_data",
            environment="scratch",
            branch_name="feature-x",
        )
    mock_conflict.assert_not_called()
    mock_register.assert_not_called()


@patch.object(_db_management, "register_database")
@patch.object(_db_management, "check_database_conflict", side_effect=ValueError("registry conflict"))
@patch.object(_db_management, "mysql_database_exists", return_value=False)
@patch.object(_db_management, "create_engine_short")
def test_registry_conflict_raises(
    mock_create_engine,
    _mock_exists,
    _mock_conflict,
    mock_register,
):
    with pytest.raises(ValueError, match="registry conflict"):
        create_database_for_environment(
            base_name="portfolio_data",
            environment="scratch",
            branch_name="feature-x",
        )
    mock_create_engine.assert_not_called()
    mock_register.assert_not_called()


@patch.object(_db_management, "register_database")
@patch.object(_db_management, "check_database_conflict")
@patch.object(_db_management, "mysql_database_exists", return_value=False)
@patch.object(_db_management, "create_engine_short")
def test_create_executes_sql_and_registers(
    mock_create_engine,
    _mock_exists,
    mock_conflict,
    mock_register,
):
    conn = MagicMock()
    mock_create_engine.return_value.begin.return_value.__enter__ = MagicMock(return_value=conn)
    mock_create_engine.return_value.begin.return_value.__exit__ = MagicMock(return_value=False)

    name = create_database_for_environment(
        base_name="portfolio_config",
        environment="prod",
        branch_name="main",
        dry_run=False,
    )
    assert name == "portfolio_config"
    conn.execute.assert_called_once()
    mock_conflict.assert_called_once_with("portfolio_config")
    mock_register.assert_called_once_with("portfolio_config", "portfolio_config", "prod", "main")
