"""
Database module for FinanceDatabase.

This module provides database access utilities with environment-aware database name resolution.
"""

# Import from SQLHelpers - commonly used functions
from .SQLHelpers import (
    DatabaseAdapter,
    get_engine,
    query_database,
    dynamic_batch_update,
    clear_table_data,
    list_tables_from_db,
    store_SQL_data_Insert_Ignore,
    ping_mysql,
    create_table_from_schema,
    execute_query,
    get_table_schema,
    store_SQL_data,
    query_database_as_dict,
    create_SQL_connection,
    close_SQL_connection,
    create_SQL_database,
)

# Import from Database - environment-aware name resolution
from .db_utils import (
    Database,  # Database name constants
    get_database_name,  # Environment-aware database name resolution
    get_environment,  # Environment detection
    clear_database_name_cache,  # Cache management
    set_environment_context,  # Environment context management
    get_current_environment,
    get_current_branch_name,
)

# Import from db_management - schema cloning, deletion, and env diff/sync
from .db_management import (
    create_database_for_environment,
    create_test_environment,
    delete_environment,
    diff_environments,
    get_protected_environments,
    list_environments,
    create_missing_databases_from_environment,
    sync_missing_tables_from_environment,
    sync_environment_from_source,
    EnvironmentDiff,
)

__all__ = [
    # SQLHelpers exports
    "DatabaseAdapter",
    "get_engine",
    "query_database",
    "dynamic_batch_update",
    "clear_table_data",
    "list_tables_from_db",
    "store_SQL_data_Insert_Ignore",
    "ping_mysql",
    "create_table_from_schema",
    "execute_query",
    "get_table_schema",
    "store_SQL_data",
    "query_database_as_dict",
    "create_SQL_connection",
    "close_SQL_connection",
    "create_SQL_database",
    # Database Utilities exports
    "Database",
    "get_database_name",
    "get_environment",
    "clear_database_name_cache",
    "set_environment_context",
    "get_current_environment",
    "get_current_branch_name",
    # Database Management exports
    "create_database_for_environment",
    "create_test_environment",
    "delete_environment",
    "diff_environments",
    "get_protected_environments",
    "list_environments",
    "create_missing_databases_from_environment",
    "sync_missing_tables_from_environment",
    "sync_environment_from_source",
    "EnvironmentDiff",
]
