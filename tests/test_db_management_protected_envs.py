"""Unit tests for DB_PROTECTED_ENVIRONMENTS and delete/list guards."""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ENV_VAR = "DB_PROTECTED_ENVIRONMENTS"


def _load_db_management_module():
    """Load db_management without importing dbase.database package __init__."""
    for name in (
        "pymysql",
        "mysql",
        "mysql.connector",
        "sqlalchemy",
        "sqlalchemy.exc",
        "dotenv",
        "trade",
        "trade.helpers",
        "trade.helpers.helper",
    ):
        sys.modules.setdefault(name, MagicMock())
    sys.modules["trade.helpers.helper"].setup_logger = lambda *a, **k: MagicMock()

    db_utils = types.ModuleType("dbase.database.db_utils")

    class Database:
        EXCLUDED_DATABASES = ["master_config"]

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

    sys.modules.pop("dbase.database.db_management", None)

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


@pytest.fixture
def db_management(monkeypatch):
    """Fresh module load with DB_PROTECTED_ENVIRONMENTS cleared."""
    monkeypatch.delenv(_ENV_VAR, raising=False)
    return _load_db_management_module()


def test_get_protected_environments_default(db_management):
    assert db_management.get_protected_environments() == []


def test_get_protected_environments_json_array(db_management, monkeypatch):
    monkeypatch.setenv(_ENV_VAR, '["prod","long_bbands_v2"]')
    assert db_management.get_protected_environments() == ["prod", "long_bbands_v2"]


def test_get_protected_environments_comma_separated(db_management, monkeypatch):
    monkeypatch.setenv(_ENV_VAR, "prod, long_bbands_v2 , scratch")
    assert db_management.get_protected_environments() == ["prod", "long_bbands_v2", "scratch"]


def test_get_protected_environments_dedupes(db_management, monkeypatch):
    monkeypatch.setenv(_ENV_VAR, "prod,prod,test")
    assert db_management.get_protected_environments() == ["prod", "test"]


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("", []),
        ("   ", []),
        ("[]", []),
        (",,", []),
        (", ,", []),
        ("scratch", ["scratch"]),
    ],
)
def test_get_protected_environments_edge_cases(db_management, monkeypatch, raw, expected):
    if raw:
        monkeypatch.setenv(_ENV_VAR, raw)
    else:
        monkeypatch.delenv(_ENV_VAR, raising=False)
    assert db_management.get_protected_environments() == expected


def test_get_protected_environments_invalid_name(db_management, monkeypatch):
    monkeypatch.setenv(_ENV_VAR, "bad;name")
    with pytest.raises(ValueError, match="invalid character"):
        db_management.get_protected_environments()


def test_get_protected_environments_invalid_json(db_management, monkeypatch):
    monkeypatch.setenv(_ENV_VAR, "[prod")
    with pytest.raises(ValueError, match=_ENV_VAR):
        db_management.get_protected_environments()


def test_delete_environment_blocks_protected(db_management, monkeypatch):
    monkeypatch.setenv(_ENV_VAR, "prod,long_bbands_v2")
    with pytest.raises(ValueError, match="long_bbands_v2") as exc:
        db_management.delete_environment(["long_bbands_v2"], confirm=True)
    assert _ENV_VAR in str(exc.value)


def test_delete_environment_not_blocked_when_protection_unset(db_management):
    with patch.object(db_management, "get_databases_for_environment", return_value={}):
        assert db_management.delete_environment(["prod"], confirm=True) == {}


def test_delete_environment_allows_prod_when_not_listed(db_management, monkeypatch):
    monkeypatch.setenv(_ENV_VAR, "long_bbands_v2")
    with patch.object(db_management, "get_databases_for_environment", return_value={}):
        assert db_management.delete_environment(["prod"], confirm=True) == {}


def _registry_environments_df():
    return pd.DataFrame(
        {"environment": ["prod", "long_bbands_v2", "test-mean-reversion", "scratch"]}
    )


def test_list_environments_exclude_prod_filters_protected(db_management, monkeypatch):
    monkeypatch.setenv(_ENV_VAR, "prod,long_bbands_v2")
    with patch.object(db_management, "get_engine", return_value=MagicMock()):
        with patch.object(
            db_management.pd,
            "read_sql",
            return_value=_registry_environments_df(),
        ):
            assert db_management.list_environments(exclude_prod=True) == [
                "scratch",
                "test-mean-reversion",
            ]


def test_list_environments_exclude_prod_false_keeps_protected(db_management, monkeypatch):
    monkeypatch.setenv(_ENV_VAR, "prod,long_bbands_v2")
    with patch.object(db_management, "get_engine", return_value=MagicMock()):
        with patch.object(
            db_management.pd,
            "read_sql",
            return_value=_registry_environments_df(),
        ):
            assert db_management.list_environments(exclude_prod=False) == [
                "long_bbands_v2",
                "prod",
                "scratch",
                "test-mean-reversion",
            ]
