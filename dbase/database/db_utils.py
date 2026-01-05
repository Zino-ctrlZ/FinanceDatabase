"""
Database name constants and environment-aware resolution.

This module provides database name constants and functions for resolving
environment-aware database names based on git branch and CLI arguments.
"""

import os


class Database:
    """Database name constants and environment-aware resolution."""

    # Base names (never change)
    PORTFOLIO_CONFIG = "portfolio_config"
    PORTFOLIO_DATA = "portfolio_data"
    STRATEGY_TRADES_SIGNALS = "strategy_trades_signals"
    PORTFOLIO_SIGNALS = "portfolio_signals"
    VOL_SURFACE = "vol_surface"
    SECURITIES_MASTER = "securities_master"
    MASTER_CONFIG = "master_config"  # Special: never suffixed

    # Exclusion list for schema cloning
    EXCLUDED_DATABASES = [
        "master_config",
        "information_schema",
        "mysql",
        "performance_schema",
        "sys",
    ]


# Module-level cache for database name resolution
_DB_NAME_CACHE = {}  # {(environment, base_name): full_name}


def get_environment(branch_name: str = None, cli_arg: str = None) -> str:
    """
    Determine environment from branch and CLI args.

    Args:
        branch_name: Git branch name from TFP-Algo
        cli_arg: CLI argument (e.g., 'mean-reversion')

    Returns:
        Environment string: 'prod', 'test', or 'test-{cli_arg}'
    """
    # Priority: CLI arg > branch detection > env var > default
    if cli_arg:
        return cli_arg

    if branch_name and branch_name != "main":
        return "test"

    return os.getenv("ENVIRONMENT", "prod")


def get_database_name(
    base_name: str, environment: str = None, branch_name: str = None
) -> str:
    """
    Get environment-aware database name with caching.

    Args:
        base_name: Base database name (e.g., 'portfolio_data')
        environment: Override environment (optional)
        branch_name: Branch name for environment detection (optional)

    Returns:
        Full database name with suffix if needed
    """
    # Special case: master_config never gets suffixed
    if base_name == Database.MASTER_CONFIG:
        return Database.MASTER_CONFIG

    # Determine environment
    if not environment:
        env = get_environment(branch_name=branch_name)
    else:
        env = environment

    # Check cache
    cache_key = (env, base_name)
    if cache_key in _DB_NAME_CACHE:
        return _DB_NAME_CACHE[cache_key]

    # Load from master_config
    if env == "prod":
        # Prod databases use base name
        db_name = base_name
    else:
        # Query master_config for test environment
        db_name = _load_database_name_from_config(base_name, env)

    # Cache and return
    _DB_NAME_CACHE[cache_key] = db_name
    return db_name


def _load_database_name_from_config(base_name: str, environment: str) -> str:
    """Load database name from master_config.database_configs."""
    # Import here to avoid circular dependency
    from .SQLHelpers import query_database

    # Query master_config (never suffixed)
    # Query by base_name and environment (branch_name not used in search)
    query = f"""
        SELECT database_name 
        FROM master_config.database_configs
        WHERE base_name = '{base_name}'
        AND environment = '{environment}'
        LIMIT 1
    """

    result = query_database("master_config", "database_configs", query)

    if result.empty:
        # Database doesn't exist for this environment
        raise ValueError(
            f"Database '{base_name}' not found for environment '{environment}'. "
            f"Create it using create_test_environment()."
        )

    return result.iloc[0]["database_name"]


def clear_database_name_cache():
    """Clear the database name cache. Useful when environment context changes."""
    global _DB_NAME_CACHE
    _DB_NAME_CACHE.clear()
