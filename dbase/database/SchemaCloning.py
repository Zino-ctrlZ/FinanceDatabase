"""
Schema cloning functions for creating test database environments.

This module provides functions to clone database schemas from production
to test environments using mysqldump and mysql restore.
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from Database import Database
from SQLHelpers import get_engine, sql_host, sql_user, sql_pw, sql_port


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
        raise ValueError(
            "Value must contain only alphanumeric characters, underscores, or hyphens"
        )


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

    return dict(zip(result["base_name"], result["database_name"]))


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
    database_name: str, base_name: str, environment: str, branch_name: str
) -> None:
    """Register a new database in master_config.database_configs."""
    # Validate all inputs
    validate_database_input(database_name)
    validate_database_input(base_name)
    validate_database_input(environment)
    if branch_name:  # branch_name can be None
        validate_database_input(branch_name)

    # Use parameterized query to prevent SQL injection
    query = text("""
        INSERT INTO master_config.database_configs 
        (database_name, base_name, environment, branch_name, created_by)
        VALUES (:database_name, :base_name, :environment, :branch_name, 'system')
    """)

    engine = get_engine("master_config")
    try:
        with engine.begin() as conn:  # Automatic transaction handling
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
        # Handle unique key constraint violation
        raise ValueError(
            f"Failed to register database '{database_name}': database may already exist. "
            f"Original error: {e}"
        ) from e


def clone_database_schema(
    source_db: str,
    target_db: str,
    *,
    schema_only: bool = True,
    include_routines: bool = True,
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
            raise ValueError(
                "Database names may not contain spaces, backticks, or semicolons."
            )

        env = os.environ.copy()
        # Avoid putting password into process args (visible in process list).
        if sql_pw:
            env["MYSQL_PWD"] = str(sql_pw)

        extra_mysqldump_args = extra_mysqldump_args or []

        # 1) Dump (schema-only by default). Use --databases to include CREATE DATABASE + USE.
        dump_cmd = [
            "mysqldump",
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

        result = subprocess.run(
            dump_cmd, capture_output=True, text=True, check=True, env=env
        )
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
            dump_sql = re.sub(
                r"DEFINER=`[^`]+`@`[^`]+`", "DEFINER=CURRENT_USER", dump_sql
            )

        # 3) Ensure the target DB exists (even if the rewritten dump creates it, this makes failures clearer).
        create_db_sql = f"CREATE DATABASE IF NOT EXISTS `{target_db}`;"
        subprocess.run(
            [
                "mysql",
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
        # Use a temp file so the operator can optionally inspect it during debugging.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False, suffix=".sql"
        ) as tf:
            tf.write(dump_sql)
            tmp_path = Path(tf.name)

        with tmp_path.open("r", encoding="utf-8") as f:
            subprocess.run(
                [
                    "mysql",
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
        # Remove temp file; comment out if you want to keep dumps for audit/debug.
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def create_test_environment(
    environment: str,
    branch_name: str,
    source_environment: str = "prod",
    exclude_databases: Optional[list[str]] = None,
    schema_only: bool = True,
) -> dict[str, str]:
    """
    Create test environment by cloning prod schemas.

    Args:
        environment: Target environment name (e.g., 'test-mean-reversion')
        branch_name: Git branch name
        source_environment: Source environment to clone from (default: 'prod')
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
    validate_database_input(source_environment)

    exclude = set(Database.EXCLUDED_DATABASES)
    if exclude_databases:
        exclude.update(exclude_databases)

    # Get source databases
    source_dbs = get_databases_for_environment(source_environment)

    created = {}
    for base_name, source_db_name in source_dbs.items():
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


def __main__():
    """CLI entry point for creating test environments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a test environment by cloning prod schemas."
    )
    parser.add_argument(
        "--env",
        required=True,
        help="Target environment name (e.g., mean-reversion => test-mean-reversion).",
    )
    parser.add_argument("--branch", required=True, help="Git branch name.")
    parser.add_argument(
        "--source-env",
        default="prod",
        help="Source environment to clone from (default: prod).",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--schema-only", action="store_true", help="Clone schema only (default)."
    )
    group.add_argument("--with-data", action="store_true", help="Clone schema + data.")

    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Additional database base names to exclude (repeatable).",
    )

    args = parser.parse_args()

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
        print(f"{base_name} -> {db_name}")


if __name__ == "__main__":
    __main__()
