"""Unit tests for environment-change callbacks in db_utils."""

from __future__ import annotations

from dbase.database import db_utils


def test_env_change_notifies_registered_callback(monkeypatch) -> None:
    """Callbacks fire when environment string changes."""

    monkeypatch.setattr(db_utils, "_ENV_CHANGE_CALLBACKS", [], raising=True)
    monkeypatch.setattr(
        db_utils,
        "ENVIRONMENT_CONTEXT",
        {"environment": "live", "branch_name": None},
        raising=True,
    )
    seen: list[tuple] = []

    def _cb(old_env, new_env) -> None:
        seen.append((old_env, new_env))

    db_utils.register_on_environment_changed(_cb)
    db_utils.set_environment_context(environment="test", branch_name="main")
    assert seen == [("live", "test")]


def test_same_env_does_not_notify(monkeypatch) -> None:
    """Callbacks are skipped when environment is unchanged."""

    monkeypatch.setattr(db_utils, "_ENV_CHANGE_CALLBACKS", [], raising=True)
    monkeypatch.setattr(
        db_utils,
        "ENVIRONMENT_CONTEXT",
        {"environment": "live", "branch_name": None},
        raising=True,
    )
    seen: list[tuple] = []

    def _cb(old_env, new_env) -> None:
        seen.append((old_env, new_env))

    db_utils.register_on_environment_changed(_cb)
    db_utils.set_environment_context(environment="live", branch_name="main")
    assert seen == []


def test_unregister_stops_notifications(monkeypatch) -> None:
    """Unregister removes the listener."""

    monkeypatch.setattr(db_utils, "_ENV_CHANGE_CALLBACKS", [], raising=True)
    monkeypatch.setattr(
        db_utils,
        "ENVIRONMENT_CONTEXT",
        {"environment": "live", "branch_name": None},
        raising=True,
    )
    seen: list[tuple] = []

    def _cb(old_env, new_env) -> None:
        seen.append((old_env, new_env))

    db_utils.register_on_environment_changed(_cb)
    db_utils.unregister_on_environment_changed(_cb)
    db_utils.set_environment_context(environment="test", branch_name=None)
    assert seen == []
