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

from .db_utils import Database
from .SQLHelpers import get_engine, sql_host, sql_user, sql_pw, sql_port
from trade.helpers.helper import setup_logger

db_management_logger = setup_logger(
    "dbase.database.db_management", stream_log_level="INFO", file_log_level="DEBUG"
)


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
                    raise ValueError(
                        f"Database '{database_name}' already exists and is active"
                    )
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
                    db_management_logger.info(
                        f"Reactivated existing database entry '{database_name}'"
                    )
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
            f"Failed to register database '{database_name}': database may already exist. "
            f"Original error: {e}"
        ) from e


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
        # Use a temp file to allow for deyb
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
        # Remove temp file, commment out to retain for debugging
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def list_environments(exclude_prod: bool = True) -> list[str]:
    """
    List all unique environments from master_config.database_configs.

    Args:
        exclude_prod: If True, filters out 'prod' environment

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
        environments = [env for env in environments if env != "prod"]

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
        raise ValueError(
            "Database names may not contain spaces, backticks, or semicolons."
        )

    env = os.environ.copy()
    # Avoid putting password into process args (visible in process list).
    if sql_pw:
        env["MYSQL_PWD"] = str(sql_pw)

    # First, check if database exists
    check_db_sql = f"SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = '{database_name}';"
    check_result = subprocess.run(
        [
            "mysql",
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
        db_management_logger.info(
            f"Database '{database_name}' does not exist in MySQL, skipping deletion"
        )
        return  # Database doesn't exist, nothing to delete

    # Drop database using mysql command-line tool
    drop_sql = f"DROP DATABASE IF EXISTS `{database_name}`;"
    try:
        subprocess.run(
            [
                "mysql",
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
        error_msg = (
            f"Failed to delete database '{database_name}' with exit code {e.returncode}"
        )
        if e.stderr:
            error_msg += f"\nmysql error: {e.stderr}"
        if e.stdout:
            error_msg += f"\nmysql output: {e.stdout}"
        raise RuntimeError(error_msg) from e

    # Verify database was actually deleted
    verify_result = subprocess.run(
        [
            "mysql",
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
        db_management_logger.debug(
            f"Failed to unregister database '{database_name}': {e}"
        )


def delete_environment(
    environments: list[str], confirm: bool = False
) -> dict[str, dict[str, str]]:
    """
    Delete all databases for one or more environments.

    Args:
        environments: List of environment names to delete
        confirm: If False, prompts for confirmation. If True, skips confirmation.

    Returns:
        Nested dict: {environment: {base_name: database_name}} of deleted databases

    Raises:
        ValueError: If any environment is 'prod' or if user cancels deletion
        RuntimeError: If database deletion fails (partial failures continue)
    """
    # Validate all environment names
    for env in environments:
        validate_database_input(env)

    # Safety check: prod can never be deleted programmatically
    if "prod" in environments:
        raise ValueError(
            "Production environment cannot be deleted programmatically. "
            "Remove 'prod' from the environments list."
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
                db_management_logger.warning(
                    f"Failed to delete database '{db_name}': {e}"
                )
                continue

    return deleted


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
    """CLI entry point for creating and deleting test environments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Manage test database environments: create or delete."
    )

    # Create subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Create environment command
    create_parser = subparsers.add_parser(
        "create", help="Create a test environment by cloning prod schemas."
    )
    create_parser.add_argument(
        "--env",
        required=True,
        help="Target environment name (e.g., mean-reversion => test-mean-reversion).",
    )
    create_parser.add_argument("--branch", required=True, help="Git branch name.")
    create_parser.add_argument(
        "--source-env",
        default="prod",
        help="Source environment to clone from (default: prod).",
    )

    group = create_parser.add_mutually_exclusive_group()
    group.add_argument(
        "--schema-only", action="store_true", help="Clone schema only (default)."
    )
    group.add_argument("--with-data", action="store_true", help="Clone schema + data.")

    create_parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Additional database base names to exclude (repeatable).",
    )

    # Delete environment command
    delete_parser = subparsers.add_parser(
        "delete", help="Delete one or more test environments."
    )
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

    elif args.command == "delete":
        try:
            deleted = delete_environment(
                environments=args.delete_env, confirm=args.confirm
            )

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
            db_management_logger.info("Available non-production environments:")
            for env in envs:
                db_management_logger.info(f"  - {env}")
        else:
            db_management_logger.info("No non-production environments found.")

    else:
        parser.print_help()


if __name__ == "__main__":
    __main__()
