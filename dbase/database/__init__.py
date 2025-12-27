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
    set_environment_context,  # NEW: Environment context management
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
from .Database import (
    Database,  # Database name constants
    get_database_name,  # Environment-aware database name resolution
    get_environment,  # Environment detection
    clear_database_name_cache,  # Cache management
)

# Import from SchemaCloning - schema cloning and deletion functions
from .SchemaCloning import (
    create_test_environment,  # Main function for creating test environments
    delete_environment,  # Main function for deleting test environments
    list_environments,  # List all non-production environments
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
    "set_environment_context",  # NEW
    "create_table_from_schema",
    "execute_query",
    "get_table_schema",
    "store_SQL_data",
    "query_database_as_dict",
    "create_SQL_connection",
    "close_SQL_connection",
    "create_SQL_database",
    # Database exports
    "Database",
    "get_database_name",
    "get_environment",
    "clear_database_name_cache",
    # SchemaCloning exports
    "create_test_environment",
    "delete_environment",
    "list_environments",
]
