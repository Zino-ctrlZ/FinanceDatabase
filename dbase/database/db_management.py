"""
Schema cloning functions for creating test database environments.

This module provides functions to clone database schemas from production
to test environments using mysqldump and mysql restore.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from .db_utils import Database
from .SQLHelpers import create_engine_short, get_engine, sql_host, sql_user, sql_pw, sql_port
from trade.helpers.helper import setup_logger

db_management_logger = setup_logger("dbase.database.db_management", stream_log_level="INFO", file_log_level="DEBUG")

BOT_CONFIG_TABLES = ("discord_channels", "bot_identity", "bot_clients")

DB_PROTECTED_ENVIRONMENTS = "DB_PROTECTED_ENVIRONMENTS"


def validate_database_input(value: str) -> None:
    """
    Validate database-related input strings to prevent SQL injection.

    Args:
        value: Value to validate

    Raises:
        ValueError: If validation fails
    """
    if not value or not isinstance(value, str):
        raise ValueError("Value must be a non-empty string")

    # Basic safety: no SQL injection characters
    dangerous_chars = ["'", '"', ";", "--", "/*", "*/", "`", "\n", "\r"]
    for char in dangerous_chars:
        if char in value:
            raise ValueError(f"Value contains invalid character: {repr(char)}")

    # Validate format: alphanumeric with underscores/hyphens only
    if not re.match(r"^[a-zA-Z0-9_-]+$", value):
        raise ValueError("Value must contain only alphanumeric characters, underscores, or hyphens")


def _parse_protected_environments_raw(raw: str) -> list[str]:
    """Parse DB_PROTECTED_ENVIRONMENTS value (JSON array or comma-separated names)."""
    raw = raw.strip()
    if not raw:
        return []

    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"{DB_PROTECTED_ENVIRONMENTS} must be a JSON array of strings, "
                f'e.g. \'["prod","long_bbands"]\': {e}'
            ) from e
        if not isinstance(parsed, list):
            raise ValueError(f"{DB_PROTECTED_ENVIRONMENTS} JSON value must be an array of strings")
        names: list[str] = []
        for item in parsed:
            if not isinstance(item, str):
                raise ValueError(f"{DB_PROTECTED_ENVIRONMENTS} JSON array must contain only strings")
            names.append(item.strip())
    else:
        names = [part.strip() for part in raw.split(",")]

    return [name for name in names if name]


def get_protected_environments() -> list[str]:
    """
    Environment names that cannot be deleted programmatically.

    Reads ``DB_PROTECTED_ENVIRONMENTS``: a JSON array (e.g. ``["prod","long_bbands"]``)
    or comma-separated list (e.g. ``prod,long_bbands``). When unset or empty after parse,
    no environments are protected. Each name is validated with :func:`validate_database_input`.
    """
    raw = os.environ.get(DB_PROTECTED_ENVIRONMENTS)
    if raw is None or not raw.strip():
        names: list[str] = []
    else:
        names = _parse_protected_environments_raw(raw)

    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        validate_database_input(name)
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def get_databases_for_environment(environment: str) -> dict[str, str]:
    """
    Get all databases for a given environment.

    Args:
        environment: Environment name (e.g., 'prod', 'test')

    Returns:
        dict: Mapping of base_name -> database_name
    """
    # Validate input
    validate_database_input(environment)

    # Use parameterized query to prevent SQL injection
    query = text("""
        SELECT base_name, database_name
        FROM master_config.database_configs
        WHERE environment = :env
        AND is_active = TRUE
    """)

    engine = get_engine("master_config")
    result = pd.read_sql(query, engine, params={"env": environment})

    if result.empty:
        return {}

    return dict(zip(result["base_name"], result["database_name"]))  # noqa


def get_tables_for_database(database_name: str) -> list[str]:
    """
    List base tables in a database by physical database name (no environment resolution).

    Args:
        database_name: Physical MySQL database name (e.g. 'portfolio_data' or 'portfolio_data_test').

    Returns:
        Sorted list of table names (BASE TABLE only, no views).
    """
    validate_database_input(database_name)

    query = text("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = :db
          AND table_type = 'BASE TABLE'
    """)
    engine = get_engine("master_config")
    result = pd.read_sql(query, engine, params={"db": database_name})

    if result.empty:
        return []

    # Use first column: MySQL may return TABLE_NAME (uppercase) depending on driver
    col = result.columns[0]
    return sorted(result[col].tolist())


@dataclass
class EnvironmentDiff:
    """
    Difference between two environments: what the target lacks vs the source.

    - missing_databases: base_name -> source database name (DBs that exist in source but not in target).
    - table_differences: for bases present in both, tables that exist in source but not in target.
    """

    source_environment: str
    target_environment: str
    missing_databases: dict[str, str]  # base_name -> source_database_name
    table_differences: dict[
        str, dict[str, Any]
    ]  # base_name -> {source_database, target_database, tables_missing_in_target}


def diff_environments(source_environment: str, target_environment: str) -> EnvironmentDiff:
    """
    Compare two environments: what databases and tables does target lack relative to source?

    Args:
        source_environment: Reference environment (e.g. 'long_bbands').
        target_environment: Environment to compare (e.g. 'test').

    Returns:
        EnvironmentDiff with missing_databases and table_differences.
    """
    validate_database_input(source_environment)
    validate_database_input(target_environment)

    source = get_databases_for_environment(source_environment)
    target = get_databases_for_environment(target_environment)

    missing_databases = {base: source_db for base, source_db in source.items() if base not in target}

    common_bases = set(source) & set(target)
    table_differences: dict[str, dict[str, Any]] = {}

    for base in common_bases:
        source_db = source[base]
        target_db = target[base]

        source_tables = set(get_tables_for_database(source_db))
        target_tables = set(get_tables_for_database(target_db))
        missing_tables = sorted(source_tables - target_tables)

        if missing_tables:
            table_differences[base] = {
                "source_database": source_db,
                "target_database": target_db,
                "tables_missing_in_target": missing_tables,
            }

    return EnvironmentDiff(
        source_environment=source_environment,
        target_environment=target_environment,
        missing_databases=missing_databases,
        table_differences=table_differences,
    )


def check_database_conflict(db_name: str) -> None:
    """Check if database name conflicts with existing database."""
    # Validate input
    validate_database_input(db_name)

    # Use parameterized query
    query = text("""
        SELECT database_name, environment, branch_name
        FROM master_config.database_configs
        WHERE database_name = :db_name
        AND is_active = TRUE
    """)

    engine = get_engine("master_config")
    result = pd.read_sql(query, engine, params={"db_name": db_name})

    if not result.empty:
        existing = result.iloc[0]
        raise ValueError(
            f"Database '{db_name}' already exists for environment '{existing['environment']}' "
            f"(branch: {existing['branch_name']}). Choose a different environment name."
        )


def register_database(
    database_name: str,
    base_name: str,
    environment: str,
    branch_name: Optional[str] = None,
) -> None:
    """Register a new database in master_config.database_configs."""
    # Validate all inputs
    validate_database_input(database_name)
    validate_database_input(base_name)
    validate_database_input(environment)
    if branch_name:  # branch_name can be None
        validate_database_input(branch_name)

    engine = get_engine("master_config")
    try:
        with engine.begin() as conn:  # Automatic transaction handling
            # Check if inactive entry exists
            check_query = text("""
                SELECT is_active FROM master_config.database_configs
                WHERE database_name = :db_name
            """)
            existing = pd.read_sql(check_query, conn, params={"db_name": database_name})

            if not existing.empty:
                # Entry exists - reactivate it if inactive, or raise error if active
                if existing.iloc[0]["is_active"]:
                    raise ValueError(f"Database '{database_name}' already exists and is active")
                else:
                    # Reactivate the existing entry
                    update_query = text("""
                        UPDATE master_config.database_configs
                        SET is_active = TRUE,
                            base_name = :base_name,
                            environment = :environment,
                            branch_name = :branch_name,
                            created_by = 'system'
                        WHERE database_name = :db_name
                    """)
                    conn.execute(
                        update_query,
                        {
                            "db_name": database_name,
                            "base_name": base_name,
                            "environment": environment,
                            "branch_name": branch_name,
                        },
                    )
                    db_management_logger.info(f"Reactivated existing database entry '{database_name}'")
                    return

            # No existing entry, insert new one
            query = text("""
                INSERT INTO master_config.database_configs 
                (database_name, base_name, environment, branch_name, created_by)
                VALUES (:database_name, :base_name, :environment, :branch_name, 'system')
            """)
            conn.execute(
                query,
                {
                    "database_name": database_name,
                    "base_name": base_name,
                    "environment": environment,
                    "branch_name": branch_name,
                },
            )
    except IntegrityError as e:
        # Handle unique key constraint violation (shouldn't happen with the check above, but just in case)
        raise ValueError(
            f"Failed to register database '{database_name}': database may already exist. Original error: {e}"
        ) from e


def _resolve_mysql_client_bin(client: str) -> str:
    """Resolve mysql or mysqldump CLI path (PATH, then common install locations)."""
    bin_path = shutil.which(client)
    if bin_path:
        return bin_path
    home = Path.home()
    for candidate in (
        f"/usr/local/bin/{client}",
        f"/usr/bin/{client}",
        f"/opt/homebrew/bin/{client}",
        str(home / "miniconda3" / "bin" / client),
        str(home / "anaconda3" / "bin" / client),
    ):
        if Path(candidate).is_file():
            return candidate
    raise FileNotFoundError(
        f"'{client}' not found on PATH or common locations. "
        f"Install MySQL client tools or add {client} to PATH."
    )


def clone_database_schema(
    source_db: str,
    target_db: str,
    *,
    schema_only: bool = True,
    include_triggers: bool = False,
    include_events: bool = False,
    strip_definers: bool = False,
    extra_mysqldump_args: Optional[list[str]] = None,
) -> None:
    """
    Clone MySQL schema (and optionally data) using mysqldump + mysql restore.

    Key properties:
      - Uses `--databases` to ensure mysqldump emits CREATE DATABASE + USE statements.
      - Rewrites those directives so restore targets `target_db`.
      - Restores using `mysql` client (not statement splitting), which is required for routines/triggers/events.
      - Does NOT drop the source DB.

    Args:
        source_db: Source database name to clone from
        target_db: Target database name to clone to
        schema_only: If True, clone schema only; if False, clone schema + data
        include_routines: Include stored procedures and functions
        include_triggers: Include triggers
        include_events: Include scheduled events
        strip_definers: Strip DEFINER clauses to prevent restore failures
        extra_mysqldump_args: Additional arguments to pass to mysqldump
    """
    # Validate inputs
    validate_database_input(source_db)
    validate_database_input(target_db)
    try:
        # Check for None values in SQL connection parameters
        if not sql_host:
            raise ValueError("MYSQL_HOST environment variable is not set")
        if not sql_user:
            raise ValueError("MYSQL_USER environment variable is not set")
        if not sql_port:
            raise ValueError("MYSQL_PORT environment variable is not set")
        # sql_pw can be None (no password), but we check it before using

        # Conservative name safety check
        illegal = re.compile(r"[`\s;]")
        if illegal.search(source_db) or illegal.search(target_db):
            raise ValueError("Database names may not contain spaces, backticks, or semicolons.")

        env = os.environ.copy()
        # Avoid putting password into process args (visible in process list).
        if sql_pw:
            env["MYSQL_PWD"] = str(sql_pw)

        extra_mysqldump_args = extra_mysqldump_args or []

        # Resolve mysqldump / mysql binaries — they may not be on the active conda env PATH.
        mysqldump_bin = _resolve_mysql_client_bin("mysqldump")
        mysql_bin = _resolve_mysql_client_bin("mysql")

        # 1) Dump (schema-only by default). Use --databases to include CREATE DATABASE + USE.
        dump_cmd = [
            mysqldump_bin,
            f"--host={sql_host}",
            f"--port={sql_port}",
            f"--user={sql_user}",
            "--databases",
            source_db,
            "--single-transaction",
            "--skip-comments",
            "--skip-add-drop-database",  # do NOT emit DROP DATABASE
            "--set-charset",
            "--skip-routines",
            *extra_mysqldump_args,
        ]

        if schema_only:
            dump_cmd.append("--no-data")

        # Include optional objects.
        if include_events:
            dump_cmd.append("--events")
        if include_triggers:
            # mysqldump includes triggers by default; add explicitly for clarity
            dump_cmd.append("--triggers")
        else:
            dump_cmd.append("--skip-triggers")

        result = subprocess.run(dump_cmd, capture_output=True, text=True, check=True, env=env)
        dump_sql = result.stdout

        # 2) Rewrite CREATE DATABASE / USE so the dump targets the new DB name.
        # Anchor to line starts to avoid accidental replacements elsewhere.
        #
        # Typical mysqldump emits lines like:
        #   CREATE DATABASE /*!32312 IF NOT EXISTS*/ `source_db` ...
        #   USE `source_db`;
        #
        # We rewrite only the database routing, not table DDL.
        dump_sql = re.sub(
            rf"(?m)^(CREATE DATABASE\b.*?`){re.escape(source_db)}(`)",
            rf"\1{target_db}\2",
            dump_sql,
        )
        dump_sql = re.sub(
            rf"(?m)^(USE `){re.escape(source_db)}(`;)",
            rf"\1{target_db}\2",
            dump_sql,
        )

        # Optional: strip DEFINER clauses to prevent restore failures if definers don't exist on target server.
        # This is common when moving dumps across environments/users.
        if strip_definers:
            dump_sql = re.sub(r"DEFINER=`[^`]+`@`[^`]+`", "DEFINER=CURRENT_USER", dump_sql)

        # 3) Ensure the target DB exists (even if the rewritten dump creates it, this makes failures clearer).
        create_db_sql = f"CREATE DATABASE IF NOT EXISTS `{target_db}`;"
        subprocess.run(
            [
                mysql_bin,
                f"--host={sql_host}",
                f"--port={sql_port}",
                f"--user={sql_user}",
                "-e",
                create_db_sql,
            ],
            check=True,
            env=env,
        )

        # 4) Restore using mysql client (stdin). This is the robust path for routines/triggers/events.
        # Use a temp file to allow for deyb
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".sql") as tf:
            tf.write(dump_sql)
            tmp_path = Path(tf.name)

        with tmp_path.open("r", encoding="utf-8") as f:
            subprocess.run(
                [
                    mysql_bin,
                    f"--host={sql_host}",
                    f"--port={sql_port}",
                    f"--user={sql_user}",
                    # no db arg here: the dump includes USE `target_db`;
                ],
                stdin=f,
                check=True,
                env=env,
                text=True,
            )
    except subprocess.CalledProcessError as e:
        error_msg = f"mysql restore failed with exit code {e.returncode}"
        if e.stderr:
            error_msg += f"\nmysql error: {e.stderr}"
        raise RuntimeError(error_msg) from e
    except Exception as e:
        raise RuntimeError(f"Error cloning database schema: {e}") from e
    finally:
        # Remove temp file, commment out to retain for debugging
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def list_environments(exclude_prod: bool = True) -> list[str]:
    """
    List all unique environments from master_config.database_configs.

    Args:
        exclude_prod: If True, filters out all protected environments (see
            :func:`get_protected_environments` / ``DB_PROTECTED_ENVIRONMENTS``).
            Parameter name kept for backward compatibility.

    Returns:
        Sorted list of environment names
    """
    # Use parameterized query to prevent SQL injection
    query = text("""
        SELECT DISTINCT environment
        FROM master_config.database_configs
        WHERE is_active = TRUE
    """)

    engine = get_engine("master_config")
    result = pd.read_sql(query, engine)

    if result.empty:
        return []

    environments = result["environment"].tolist()

    if exclude_prod:
        protected = set(get_protected_environments())
        environments = [env for env in environments if env not in protected]

    return sorted(environments)


def delete_database(database_name: str) -> None:
    """
    Drop a single MySQL database using mysql command-line tool.

    Args:
        database_name: Name of the database to delete

    Raises:
        ValueError: If database name is invalid or in excluded list
        RuntimeError: If mysql command fails or database still exists after deletion attempt
    """
    # Validate input
    validate_database_input(database_name)

    # Check that database is not in excluded list
    if database_name in Database.EXCLUDED_DATABASES:
        raise ValueError(f"Cannot delete protected database: {database_name}")

    # Check for None values in SQL connection parameters
    if not sql_host:
        raise ValueError("MYSQL_HOST environment variable is not set")
    if not sql_user:
        raise ValueError("MYSQL_USER environment variable is not set")
    if not sql_port:
        raise ValueError("MYSQL_PORT environment variable is not set")

    # Conservative name safety check
    illegal = re.compile(r"[`\s;]")
    if illegal.search(database_name):
        raise ValueError("Database names may not contain spaces, backticks, or semicolons.")

    env = os.environ.copy()
    # Avoid putting password into process args (visible in process list).
    if sql_pw:
        env["MYSQL_PWD"] = str(sql_pw)

    mysql_bin = _resolve_mysql_client_bin("mysql")

    # First, check if database exists
    check_db_sql = f"SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = '{database_name}';"
    check_result = subprocess.run(
        [
            mysql_bin,
            f"--host={sql_host}",
            f"--port={sql_port}",
            f"--user={sql_user}",
            "-e",
            check_db_sql,
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )

    db_exists = database_name in check_result.stdout
    if not db_exists:
        db_management_logger.info(f"Database '{database_name}' does not exist in MySQL, skipping deletion")
        return  # Database doesn't exist, nothing to delete

    # Drop database using mysql command-line tool
    drop_sql = f"DROP DATABASE IF EXISTS `{database_name}`;"
    try:
        subprocess.run(
            [
                mysql_bin,
                f"--host={sql_host}",
                f"--port={sql_port}",
                f"--user={sql_user}",
                "-e",
                drop_sql,
            ],
            check=True,
            env=env,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to delete database '{database_name}' with exit code {e.returncode}"
        if e.stderr:
            error_msg += f"\nmysql error: {e.stderr}"
        if e.stdout:
            error_msg += f"\nmysql output: {e.stdout}"
        raise RuntimeError(error_msg) from e

    # Verify database was actually deleted
    verify_result = subprocess.run(
        [
            mysql_bin,
            f"--host={sql_host}",
            f"--port={sql_port}",
            f"--user={sql_user}",
            "-e",
            check_db_sql,
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )

    if database_name in verify_result.stdout:
        # Database still exists after DROP command
        raise RuntimeError(
            f"Database '{database_name}' still exists after deletion attempt. "
            f"This may be due to active connections or locks. "
            f"Please check MySQL connections and try again."
        )

    db_management_logger.info(f"Successfully deleted database '{database_name}'")


def unregister_database(database_name: str) -> None:
    """
    Soft delete database entry from master_config.database_configs.

    Sets is_active = FALSE rather than hard deleting for audit trail.
    Handles case where database doesn't exist in config gracefully.

    Args:
        database_name: Name of the database to unregister
    """
    # Validate input
    validate_database_input(database_name)

    # Use parameterized query to prevent SQL injection
    query = text("""
        UPDATE master_config.database_configs
        SET is_active = FALSE
        WHERE database_name = :db_name
    """)

    engine = get_engine("master_config")
    try:
        with engine.begin() as conn:  # Automatic transaction handling
            conn.execute(query, {"db_name": database_name})
            # If no rows were updated, database doesn't exist in config - that's OK
            # We just skip silently (orphaned database case)
    except Exception as e:
        # Log but don't fail - database might already be unregistered
        # This is a soft failure case
        db_management_logger.debug(f"Failed to unregister database '{database_name}': {e}")


def delete_environment(environments: list[str], confirm: bool = False) -> dict[str, dict[str, str]]:
    """
    Delete all databases for one or more environments.

    Args:
        environments: List of environment names to delete
        confirm: If False, prompts for confirmation. If True, skips confirmation.

    Returns:
        Nested dict: {environment: {base_name: database_name}} of deleted databases

    Raises:
        ValueError: If any environment is protected (see ``DB_PROTECTED_ENVIRONMENTS``)
            or if user cancels deletion
        RuntimeError: If database deletion fails (partial failures continue)
    """
    # Validate all environment names
    for env in environments:
        validate_database_input(env)

    protected = set(get_protected_environments())
    blocked = [env for env in environments if env in protected]
    if blocked:
        blocked_str = ", ".join(blocked)
        raise ValueError(
            f"Cannot delete protected environment(s): {blocked_str}. "
            f"Remove them from the environments list or adjust {DB_PROTECTED_ENVIRONMENTS}."
        )

    # Collect all databases across all environments
    all_databases = {}
    for env in environments:
        dbs = get_databases_for_environment(env)
        if dbs:
            all_databases[env] = dbs

    # If no databases found, return empty dict
    if not all_databases:
        return {}

    # Confirmation flow (if confirm=False)
    if not confirm:
        # Display databases grouped by environment
        db_management_logger.info("\nThe following databases will be deleted:")
        db_management_logger.info("=" * 60)
        for env, dbs in all_databases.items():
            db_management_logger.info(f"\nEnvironment: {env}")
            for base_name, db_name in dbs.items():
                db_management_logger.info(f"  - {base_name} -> {db_name}")
        db_management_logger.info("=" * 60)

        # Prompt for confirmation
        while True:
            response = input("\nDelete these databases? (y/n): ").strip().lower()
            if response in ("n", "no"):
                raise ValueError("Deletion cancelled by user")
            elif response in ("y", "yes"):
                break
            else:
                print("Please enter 'y' or 'n'")

    # Delete databases for each environment
    deleted = {}
    for env, dbs in all_databases.items():
        deleted[env] = {}
        for base_name, db_name in dbs.items():
            try:
                db_management_logger.info(f"Deleting database '{db_name}'")
                # Delete MySQL database
                delete_database(db_name)
                # Unregister from config (soft delete)
                unregister_database(db_name)
                deleted[env][base_name] = db_name
            except Exception as e:
                # Continue on partial failures
                db_management_logger.warning(f"Failed to delete database '{db_name}': {e}")
                continue

    return deleted


def create_missing_databases_from_environment(
    source_environment: str,
    target_environment: str,
    branch_name: Optional[str] = None,
    schema_only: bool = True,
    apply: bool = False,
) -> dict[str, Any]:
    """
    Create in target environment any databases that exist in source but not in target.

    Args:
        source_environment: Environment to clone from (e.g. 'long_bbands').
        target_environment: Environment to add DBs to (e.g. 'test').
        branch_name: Optional branch name for registration (stored in master_config).
        schema_only: If True, clone schema only; if False, clone schema + data.
        apply: If False (default), dry run: return structure with no changes. If True, perform creates.

    Returns:
        dict with keys: created (base_name -> target_db_name), failed (base_name -> error message), dry_run (bool).
    """
    validate_database_input(source_environment)
    validate_database_input(target_environment)
    if branch_name:
        validate_database_input(branch_name)

    diff = diff_environments(source_environment, target_environment)
    created: dict[str, str] = {}
    failed: dict[str, str] = {}

    if not apply:
        return {"created": created, "failed": failed, "dry_run": True}

    for base_name, source_db in diff.missing_databases.items():
        target_db_name = f"{base_name}_{target_environment}"
        try:
            check_database_conflict(target_db_name)
            clone_database_schema(source_db, target_db_name, schema_only=schema_only)
            register_database(target_db_name, base_name, target_environment, branch_name)
            created[base_name] = target_db_name
        except Exception as e:
            failed[base_name] = str(e)
            db_management_logger.warning(f"Failed to create database '{target_db_name}' from '{source_db}': {e}")

    return {"created": created, "failed": failed, "dry_run": False}


def sync_missing_tables_between_databases(
    source_db: str,
    target_db: str,
    tables: list[str],
    copy_data: bool = False,
    apply: bool = False,
) -> dict[str, Any]:
    """
    Create missing tables in target DB from source (CREATE TABLE ... LIKE; optionally INSERT).

    Args:
        source_db: Source physical database name.
        target_db: Target physical database name.
        tables: List of table names to create in target (and optionally copy data).
        copy_data: If True, copy data into new tables; default False (schema only).
        apply: If False (default), dry run. If True, perform creates.

    Returns:
        dict with keys: synced (list of table names), failed (table_name -> error message), dry_run (bool).
    """
    validate_database_input(source_db)
    validate_database_input(target_db)
    for tbl in tables:
        validate_database_input(tbl)

    synced: list[str] = []
    failed: dict[str, str] = {}

    if not apply:
        return {"synced": synced, "failed": failed, "dry_run": True}

    engine = get_engine("master_config")
    for tbl in tables:
        try:
            with engine.begin() as conn:
                # CREATE TABLE target.tbl LIKE source.tbl (identifiers validated above)
                conn.execute(text(f"CREATE TABLE IF NOT EXISTS `{target_db}`.`{tbl}` LIKE `{source_db}`.`{tbl}`"))
                if copy_data:
                    conn.execute(text(f"INSERT INTO `{target_db}`.`{tbl}` SELECT * FROM `{source_db}`.`{tbl}`"))
            synced.append(tbl)
        except Exception as e:
            failed[tbl] = str(e)
            db_management_logger.warning(f"Failed to sync table '{tbl}' from {source_db} to {target_db}: {e}")

    return {"synced": synced, "failed": failed, "dry_run": False}


def sync_missing_tables_from_environment(
    source_environment: str,
    target_environment: str,
    copy_data: bool = False,
    apply: bool = False,
) -> dict[str, Any]:
    """
    Add missing tables in target env from source (for DBs that exist in both).

    Args:
        source_environment: Source environment (e.g. 'long_bbands').
        target_environment: Target environment (e.g. 'test').
        copy_data: If True, copy data into new tables; default False.
        apply: If False (default), dry run. If True, perform sync.

    Returns:
        dict with keys: synced_tables (base_name -> list of table names),
        failed_tables (base_name -> {table_name: error}), dry_run (bool).
    """
    validate_database_input(source_environment)
    validate_database_input(target_environment)

    diff = diff_environments(source_environment, target_environment)
    synced_tables: dict[str, list[str]] = {}
    failed_tables: dict[str, dict[str, str]] = {}

    if not apply:
        return {
            "synced_tables": synced_tables,
            "failed_tables": failed_tables,
            "dry_run": True,
        }

    for base_name, details in diff.table_differences.items():
        source_db = details["source_database"]
        target_db = details["target_database"]
        missing = details["tables_missing_in_target"]

        result = sync_missing_tables_between_databases(source_db, target_db, missing, copy_data=copy_data, apply=True)
        synced_tables[base_name] = result["synced"]
        if result["failed"]:
            failed_tables[base_name] = result["failed"]

    return {
        "synced_tables": synced_tables,
        "failed_tables": failed_tables,
        "dry_run": False,
    }


def sync_environment_from_source(
    source_environment: str,
    target_environment: str,
    branch_name: Optional[str] = None,
    schema_only: bool = True,
    copy_table_data: bool = False,
    sync_databases: bool = True,
    sync_tables: bool = True,
    apply: bool = False,
) -> dict[str, Any]:
    """
    One-shot: diff then optionally create missing DBs and sync missing tables in target from source.

    Args:
        source_environment: Source env (e.g. 'long_bbands').
        target_environment: Target env (e.g. 'test').
        branch_name: Optional branch name for new DB registration (stored in master_config).
        schema_only: For new DBs, clone schema only (default True).
        copy_table_data: For new tables, copy data (default False).
        sync_databases: If True, create missing databases when apply=True.
        sync_tables: If True, add missing tables when apply=True.
        apply: If False (default), dry run; no changes. If True, perform sync.

    Returns:
        dict with: diff (EnvironmentDiff), created_databases, failed_databases,
        synced_tables, failed_tables, dry_run.
    """
    validate_database_input(source_environment)
    validate_database_input(target_environment)
    if branch_name:
        validate_database_input(branch_name)

    diff = diff_environments(source_environment, target_environment)
    result: dict[str, Any] = {
        "diff": diff,
        "created_databases": {},
        "failed_databases": {},
        "synced_tables": {},
        "failed_tables": {},
        "dry_run": not apply,
    }

    if not apply:
        return result

    if sync_databases and diff.missing_databases:
        db_result = create_missing_databases_from_environment(
            source_environment,
            target_environment,
            branch_name=branch_name,
            schema_only=schema_only,
            apply=True,
        )
        result["created_databases"] = db_result["created"]
        result["failed_databases"] = db_result["failed"]

    if sync_tables and diff.table_differences:
        tbl_result = sync_missing_tables_from_environment(
            source_environment,
            target_environment,
            copy_data=copy_table_data,
            apply=True,
        )
        result["synced_tables"] = tbl_result["synced_tables"]
        result["failed_tables"] = tbl_result["failed_tables"]

    return result


def bot_config_table_row_counts(portfolio_config_db: str) -> dict[str, int]:
    """
    Return row counts for bot config tables in a physical portfolio_config database.

    Args:
        portfolio_config_db: Physical MySQL database name (e.g. portfolio_config_scratch).

    Returns:
        Mapping of table name -> row count for discord_channels, bot_identity, bot_clients.
    """
    validate_database_input(portfolio_config_db)
    counts: dict[str, int] = {}
    engine = create_engine_short(portfolio_config_db)
    with engine.connect() as conn:
        for table in BOT_CONFIG_TABLES:
            result = conn.execute(text(f"SELECT COUNT(*) AS n FROM `{table}`"))
            counts[table] = int(result.scalar() or 0)
    return counts


def log_empty_bot_config_hint_after_create(
    *,
    target_environment: str,
    source_environment: str,
    created: dict[str, str],
    schema_only: bool,
) -> None:
    """
    After schema-only create, log seed/verify commands when bot tables exist but have 0 rows.

    Informational only; failures are logged at DEBUG and do not affect create.
    """
    if not schema_only:
        return
    portfolio_config_db = created.get(Database.PORTFOLIO_CONFIG)
    if not portfolio_config_db:
        return
    try:
        counts = bot_config_table_row_counts(portfolio_config_db)
        if any(counts.values()):
            return
        db_management_logger.info(
            "portfolio_config bot tables are empty (schema-only create). "
            "Seed bot config manually, then verify:\n"
            f"  python -m algo.tools.seed_bot_config --env {target_environment} "
            f"--from-source-env {source_environment}\n"
            f"  # or: --from-yaml --persona tinubu|emefiele\n"
            f"  python -m algo.tools.verify_bot_config --env {target_environment}"
        )
    except Exception as e:
        db_management_logger.debug("Could not log empty bot config hint: %s", e)


def resolve_physical_database_name(base_name: str, environment: str) -> str:
    """Map base_name + environment to the physical MySQL database name."""
    validate_database_input(base_name)
    validate_database_input(environment)
    if environment == "prod":
        return base_name
    return f"{base_name}_{environment}"


def mysql_database_exists(database_name: str) -> bool:
    """Return True if a schema with this name exists in MySQL."""
    validate_database_input(database_name)
    query = text("""
        SELECT SCHEMA_NAME
        FROM information_schema.SCHEMATA
        WHERE SCHEMA_NAME = :db_name
    """)
    engine = get_engine("master_config")
    result = pd.read_sql(query, engine, params={"db_name": database_name})
    return not result.empty


def _resolve_branch_name_cli(branch: Optional[str]) -> str:
    if branch:
        validate_database_input(branch)
        return branch
    from trade.helpers.git import git_get_current_branch

    repo_path = os.environ.get("ALGO_DIR", ".")
    return git_get_current_branch(repo_path)


def create_database_for_environment(
    *,
    base_name: str,
    environment: str,
    branch_name: Optional[str] = None,
    dry_run: bool = False,
) -> str:
    """
    Create an empty MySQL database and register it in master_config.database_configs.

    Does not clone schema from another environment.

    Returns:
        Physical database_name created (or would create when dry_run=True).
    """
    validate_database_input(base_name)
    validate_database_input(environment)
    if branch_name:
        validate_database_input(branch_name)

    if base_name in Database.EXCLUDED_DATABASES:
        raise ValueError(f"Cannot create protected database base name: {base_name}")

    database_name = resolve_physical_database_name(base_name, environment)

    if dry_run:
        db_management_logger.info(
            "Dry run: would CREATE DATABASE `%s` DEFAULT CHARACTER SET utf8; "
            "register base_name=%r environment=%r branch_name=%r",
            database_name,
            base_name,
            environment,
            branch_name,
        )
        return database_name

    if mysql_database_exists(database_name):
        raise ValueError(f"MySQL database '{database_name}' already exists")

    check_database_conflict(database_name)

    engine = create_engine_short("master_config")
    create_sql = text(f"CREATE DATABASE IF NOT EXISTS `{database_name}` DEFAULT CHARACTER SET utf8")
    with engine.begin() as conn:
        conn.execute(create_sql)

    register_database(database_name, base_name, environment, branch_name)
    db_management_logger.info(
        "Created database '%s' (base_name=%r, environment=%r)",
        database_name,
        base_name,
        environment,
    )
    return database_name


def create_test_environment(
    environment: str,
    branch_name: str,
    source_environment: str,
    exclude_databases: Optional[list[str]] = None,
    schema_only: bool = True,
) -> dict[str, str]:
    """
    Create an environment by cloning schemas (and optionally data) from a source environment.

    Args:
        environment: Target environment name (e.g., 'test-mean-reversion')
        branch_name: Git branch name
        source_environment: Source environment to clone from (required; e.g. long_bbands_v2)
        exclude_databases: Additional databases to exclude
        schema_only: If True, clone schema only; if False, clone schema + data

    Returns:
        dict: Created database names {base_name: full_name}

    Note on Transaction Handling:
    - Partial failures are unlikely but possible (network issues, disk full, etc.)
    - Current approach: Continue on partial failure. Each database is independent,
      so partial success is acceptable. Manual cleanup can remove orphaned databases.
    """
    # Validate inputs
    validate_database_input(environment)
    if branch_name:
        validate_database_input(branch_name)
    if not source_environment:
        raise ValueError("source_environment is required")
    validate_database_input(source_environment)

    exclude = set(Database.EXCLUDED_DATABASES)
    if exclude_databases:
        exclude.update(exclude_databases)

    # Get source databases
    source_dbs = get_databases_for_environment(source_environment)

    created = {}
    for base_name, source_db_name in source_dbs.items():
        print(f"Processing base database '{base_name}' for environment '{environment}'...")
        if base_name in exclude:
            continue

        target_db_name = f"{base_name}_{environment}"

        # Check for conflicts
        check_database_conflict(target_db_name)

        # Clone schema (and optionally data)
        clone_database_schema(source_db_name, target_db_name, schema_only=schema_only)

        # Register in master_config
        register_database(target_db_name, base_name, environment, branch_name)

        created[base_name] = target_db_name

    return created


def build_cli_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser (used by __main__ and tests)."""
    parser = argparse.ArgumentParser(description="Manage test database environments: create or delete.")

    # Create subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Create environment command
    create_parser = subparsers.add_parser(
        "create",
        help="Create an environment by cloning schemas/data from a required source environment.",
    )
    create_parser.add_argument(
        "--env",
        required=True,
        help="Target environment name (e.g., mean-reversion => test-mean-reversion).",
    )
    create_parser.add_argument("--branch", required=True, help="Git branch name.")
    create_parser.add_argument(
        "--source-env",
        required=True,
        help="Source environment to clone from (required; e.g. long_bbands_v2).",
    )

    group = create_parser.add_mutually_exclusive_group()
    group.add_argument("--schema-only", action="store_true", help="Clone schema only (default).")
    group.add_argument("--with-data", action="store_true", help="Clone schema + data.")

    create_parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Additional database base names to exclude (repeatable).",
    )

    # Delete environment command
    delete_parser = subparsers.add_parser("delete", help="Delete one or more test environments.")
    delete_parser.add_argument(
        "--delete-env",
        nargs="+",
        required=True,
        help="Environment name(s) to delete (e.g., test-mean-reversion).",
    )
    delete_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip confirmation prompt and proceed with deletion.",
    )

    subparsers.add_parser("list", help="List non-production environments.")

    # Diff command: compare two environments
    diff_parser = subparsers.add_parser("diff", help="Show what target env is missing vs source (DBs and tables).")
    diff_parser.add_argument(
        "--source-env",
        required=True,
        help="Source environment (e.g. long_bbands).",
    )
    diff_parser.add_argument(
        "--target-env",
        required=True,
        help="Target environment (e.g. test).",
    )

    # Sync command: create missing DBs and tables in target from source (dry run unless --apply)
    sync_parser = subparsers.add_parser(
        "sync",
        help="Create missing DBs and tables in target from source. Dry run unless --apply.",
    )
    sync_parser.add_argument("--source-env", required=True, help="Source environment.")
    sync_parser.add_argument("--target-env", required=True, help="Target environment.")
    sync_parser.add_argument(
        "--branch",
        required=False,
        default=None,
        help="Optional branch name for new DB registration (stored in master_config).",
    )
    sync_parser.add_argument(
        "--with-data",
        action="store_true",
        help="Copy data for new tables and new DBs (default: schema only).",
    )
    sync_parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes. Default is dry run.",
    )

    create_db_parser = subparsers.add_parser(
        "create-db",
        help="Create an empty database and register it in master_config (no schema clone).",
    )
    create_db_parser.add_argument("--env", required=True, help="Environment name (e.g. scratch, prod).")
    create_db_parser.add_argument(
        "--name",
        required=True,
        help="Base database name (e.g. portfolio_config, portfolio_data).",
    )
    create_db_parser.add_argument(
        "--branch",
        default=None,
        help="Git branch for registry (default: current branch from ALGO_DIR or '.').",
    )
    create_db_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log planned CREATE and registry only; do not apply.",
    )

    return parser


def __main__():
    """CLI entry point for creating and deleting test environments."""
    parser = build_cli_parser()
    args = parser.parse_args()

    if args.command == "create":
        # Default behavior: schema only unless --with-data is passed.
        schema_only = True
        if args.with_data:
            schema_only = False
        elif args.schema_only:
            schema_only = True

        created = create_test_environment(
            environment=args.env,
            branch_name=args.branch,
            source_environment=args.source_env,
            exclude_databases=args.exclude or None,
            schema_only=schema_only,
        )

        # Print in a deterministic / machine-readable way
        for base_name, db_name in created.items():
            db_management_logger.info(f"{base_name} -> {db_name}")

        log_empty_bot_config_hint_after_create(
            target_environment=args.env,
            source_environment=args.source_env,
            created=created,
            schema_only=schema_only,
        )

    elif args.command == "delete":
        try:
            deleted = delete_environment(environments=args.delete_env, confirm=args.confirm)

            # Print results
            if deleted:
                db_management_logger.info("\nSuccessfully deleted databases:")
                for env, dbs in deleted.items():
                    db_management_logger.info(f"\nEnvironment: {env}")
                    for base_name, db_name in dbs.items():
                        db_management_logger.info(f"  - {base_name} -> {db_name}")
            else:
                db_management_logger.info("No databases found to delete.")
        except ValueError as e:
            db_management_logger.error(f"Error: {e}")
            exit(1)
        except Exception as e:
            db_management_logger.error(f"Unexpected error: {e}", exc_info=True)
            exit(1)

    elif args.command == "list":
        envs = list_environments(exclude_prod=True)
        if envs:
            db_management_logger.info(
                "Available environments (protected excluded; see %s):",
                DB_PROTECTED_ENVIRONMENTS,
            )
            for env in envs:
                db_management_logger.info(f"  - {env}")
        else:
            db_management_logger.info(
                "No environments found (protected excluded; see %s).",
                DB_PROTECTED_ENVIRONMENTS,
            )

    elif args.command == "diff":
        diff = diff_environments(args.source_env, args.target_env)
        db_management_logger.info(f"Diff source={diff.source_environment} target={diff.target_environment}")
        db_management_logger.info("Missing databases (base -> source DB): %s", diff.missing_databases)
        for base, details in diff.table_differences.items():
            db_management_logger.info(
                "  %s: tables missing in target: %s",
                base,
                details["tables_missing_in_target"],
            )

    elif args.command == "create-db":
        branch = _resolve_branch_name_cli(args.branch)
        database_name = create_database_for_environment(
            base_name=args.name,
            environment=args.env,
            branch_name=branch,
            dry_run=args.dry_run,
        )
        db_management_logger.info("%s -> %s", args.name, database_name)

    elif args.command == "sync":
        if not args.apply:
            db_management_logger.info("Dry run; use --apply to apply changes.")
        result = sync_environment_from_source(
            source_environment=args.source_env,
            target_environment=args.target_env,
            branch_name=args.branch,
            schema_only=not args.with_data,
            copy_table_data=args.with_data,
            sync_databases=True,
            sync_tables=True,
            apply=args.apply,
        )
        db_management_logger.info("dry_run: %s", result["dry_run"])
        db_management_logger.info("created_databases: %s", result["created_databases"])
        db_management_logger.info("failed_databases: %s", result["failed_databases"])
        db_management_logger.info("synced_tables: %s", result["synced_tables"])
        db_management_logger.info("failed_tables: %s", result["failed_tables"])
        db_management_logger.info("diff: %s", result["diff"])

    else:
        parser.print_help()


if __name__ == "__main__":
    __main__()
