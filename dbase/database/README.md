# Database Module

## Overview

The `database` module provides utilities for interacting with MySQL databases using SQLAlchemy, with support for environment-aware database name resolution. This allows the same codebase to work seamlessly across production and test environments by automatically resolving database names based on the current environment context.

## Environment Management System

The module includes a comprehensive environment management system that:

- Automatically suffixes database names based on git branch and environment
- Tracks all database environments in `master_config.database_configs` table
- Clones production schemas to test environments using mysqldump
- Detects environment from TFP-Algo git branch and CLI arguments

### Environment Detection

Environments are determined by:

1. **CLI argument** (`--env`): Highest priority - creates `test-{cli_arg}` environment
2. **Git branch**: `main` branch = `prod`, other branches = `test`
3. **Environment variable**: `ENVIRONMENT` env var (fallback)
4. **Default**: `prod` if none of the above

### Database Name Resolution

Database names are resolved automatically based on the environment context:

- **Production**: Uses base database names (e.g., `portfolio_data`)
- **Test environments**: Uses suffixed names (e.g., `portfolio_data_test` or `portfolio_data_test-mean-reversion`)

The resolution is cached for performance and automatically clears when the environment context changes.

## Modules

### SQLHelpers.py

Core database access utilities with environment-aware resolution.

#### Key Functions

**`get_engine(db_name)`**

- Creates a SQLAlchemy engine with environment-aware database name resolution
- Automatically resolves `db_name` based on current environment context
- Caches engines per process for performance

**`set_environment_context(environment, branch_name)`**

- Sets the environment context for database name resolution
- Called by TFP-Algo runner.py at startup
- Clears database name cache when context changes

**`query_database(db, tbl_name, query)`**

- Executes SQL queries with environment-aware database resolution
- Returns pandas DataFrame

**`DatabaseAdapter`**

- High-level database access class
- Methods automatically use environment-aware database names:
  - `save_to_database()`: Save data with environment-aware resolution
  - `query_database()`: Query with automatic database name rewriting

**`dynamic_batch_update(db, table_name, update_values, condition)`**

- Updates records in batches with environment-aware database resolution

### Database.py

Database name constants and environment-aware resolution.

#### Database Class

Constants for database base names:

- `PORTFOLIO_CONFIG`
- `PORTFOLIO_DATA`
- `STRATEGY_TRADES_SIGNALS`
- `PORTFOLIO_SIGNALS`
- `VOL_SURFACE`
- `SECURITIES_MASTER`
- `MASTER_CONFIG` (never suffixed)

#### Key Functions

**`get_environment(branch_name, cli_arg)`**

- Determines environment from branch and CLI args
- Returns: `'prod'`, `'test'`, or `'test-{cli_arg}'`

**`get_database_name(base_name, environment, branch_name)`**

- Resolves environment-aware database name with caching
- For production: returns base name
- For test: queries `master_config.database_configs` for environment-specific name

**`clear_database_name_cache()`**

- Clears the database name resolution cache

### SchemaCloning.py

Schema cloning functions for creating test database environments.

#### Key Functions

**`create_test_environment(environment, branch_name, source_environment, exclude_databases, schema_only)`**

- Main function for creating test environments by cloning production schemas
- Clones all databases from source environment (default: `prod`)
- Registers new databases in `master_config.database_configs`
- Returns dict mapping `{base_name: full_name}`

**`clone_database_schema(source_db, target_db, schema_only, include_routines, ...)`**

- Low-level function to clone a single database schema
- Uses mysqldump and mysql restore
- Supports schema-only or full data cloning

**`delete_environment(environments, confirm)`**

- Main function for deleting test environments
- Deletes all databases for one or more environments
- Requires confirmation by default (unless `confirm=True`)
- Production environments are protected and cannot be deleted programmatically
- Returns nested dict: `{environment: {base_name: database_name}}`

**`list_environments(exclude_prod)`**

- Lists all unique environments from `master_config.database_configs`
- By default excludes production environment
- Returns sorted list of environment names

## Usage Examples

### Basic Usage with Environment-Aware Resolution

```python
from dbase.database import get_engine, query_database, DatabaseAdapter

# Environment context is set automatically by TFP-Algo runner.py
# All database operations automatically use environment-aware names

# Get engine (automatically resolves to test DB if in test environment)
engine = get_engine('portfolio_data')

# Query database (automatically uses correct environment DB)
df = query_database('portfolio_data', 'trades', 'SELECT * FROM trades LIMIT 10')

# Use DatabaseAdapter (automatically environment-aware)
adapter = DatabaseAdapter()
adapter.save_to_database(data, 'portfolio_data', 'trades')
```

### Setting Environment Context Manually

```python
from dbase.database import set_environment_context, get_database_name

# Set environment context (usually done by runner.py)
set_environment_context(environment='test-mean-reversion', branch_name='feature-branch')

# Get resolved database name
db_name = get_database_name('portfolio_data')  # Returns 'portfolio_data_test-mean-reversion'
```

### Creating Test Environments

```python
from dbase.database import create_test_environment

# Create a test environment by cloning production schemas
created = create_test_environment(
    environment='test-mean-reversion',
    branch_name='feature-branch',
    source_environment='prod',
    schema_only=True  # Clone schema only, no data
)

# Returns: {'portfolio_data': 'portfolio_data_test-mean-reversion', ...}
```

### Deleting Test Environments

```python
from dbase.database import delete_environment, list_environments

# List available test environments
envs = list_environments()
print(envs)  # ['test', 'test-mean-reversion', 'test-arbitrage']

# Delete single environment (shows databases, prompts for confirmation)
deleted = delete_environment(['test-mean-reversion'], confirm=False)
# Shows: "The following databases will be deleted: ..."
# Prompts: "Delete these databases? (y/n): "
# Returns: {'test-mean-reversion': {'portfolio_data': 'portfolio_data_test-mean-reversion', ...}}

# Delete multiple environments (skips confirmation)
deleted = delete_environment(['test-mean-reversion', 'test-arbitrage'], confirm=True)
# Returns: {
#   'test-mean-reversion': {'portfolio_data': 'portfolio_data_test-mean-reversion', ...},
#   'test-arbitrage': {'portfolio_data': 'portfolio_data_test-arbitrage', ...}
# }

# Production protection: this will raise ValueError
# delete_environment(['prod'])  # ERROR: Production cannot be deleted
```

### Using Database Constants

```python
from dbase.database import Database, get_database_name

# Use constants for database names
db_name = get_database_name(Database.PORTFOLIO_DATA)
# In prod: 'portfolio_data'
# In test: 'portfolio_data_test' (or environment-specific)
```

## Architecture

### Environment Context Flow

1. TFP-Algo `runner.py` detects environment from git branch and CLI args
2. Calls `set_environment_context()` to set module-level context
3. All database operations use `get_database_name()` to resolve names
4. Resolution is cached for performance

### Database Name Resolution

```
Base Name: 'portfolio_data'
  ↓
Environment Context: 'test-mean-reversion'
  ↓
Query master_config.database_configs
  ↓
Resolved Name: 'portfolio_data_test-mean-reversion'
```

### Schema Cloning Process

1. Query `master_config.database_configs` for source environment databases
2. For each database:
   - Check for name conflicts
   - Clone schema using mysqldump
   - Register in `master_config.database_configs`
3. Return mapping of created databases

### Environment Deletion Process

1. Validate all environment names (production protection)
2. Collect all databases for each environment
3. Display databases grouped by environment (if confirmation required)
4. Prompt for user confirmation (`y/n`)
5. For each database:
   - Delete MySQL database using `DROP DATABASE IF EXISTS`
   - Soft delete from `master_config.database_configs` (set `is_active = FALSE`)
6. Return mapping of deleted databases

## CLI Usage

The `SchemaCloning.py` module provides a command-line interface for managing test environments:

### Creating Test Environments

```bash
# Create a test environment
python dbase/database/SchemaCloning.py create \
    --env mean-reversion \
    --branch feature-branch \
    --schema-only

# Create with data
python dbase/database/SchemaCloning.py create \
    --env mean-reversion \
    --branch feature-branch \
    --with-data
```

### Listing Environments

```bash
# List all non-production environments
python dbase/database/db_management.py list
```

### Deleting Test Environments

```bash
# Delete single environment (shows databases, asks for confirmation)
python dbase/database/db_management.py delete --delete-env test-mean-reversion

# Delete multiple environments (shows all databases, asks for confirmation)
python dbase/database/db_management.py delete \
    --delete-env test-mean-reversion test-arbitrage

# Delete without confirmation prompt
python dbase/database/db_management.py delete \
    --delete-env test-mean-reversion \
    --confirm
```

**Safety Notes:**

- Production environments (`prod`) cannot be deleted programmatically
- Confirmation is required by default (use `--confirm` to skip)
- Entering `'n'` cancels the entire operation - no databases are deleted

## Configuration

### Required Environment Variables

- `MYSQL_HOST`: MySQL server host
- `MYSQL_PORT`: MySQL server port
- `MYSQL_USER`: MySQL username
- `MYSQL_PASSWORD`: MySQL password
- `ALGO_DIR`: Path to TFP-Algo repository (for branch detection)

### Database Setup

The `master_config` database and `database_configs` table must be created manually before using the environment management system. See the implementation plan for SQL setup scripts.

## Notes

- **Production databases**: Always use base names (never suffixed)
- **Test databases**: Use pattern `{base_name}_{environment}`
- **master_config**: Special database that is never suffixed
- **Caching**: Database name resolution is cached per environment/base_name combination
- **Partial failures**: Schema cloning and deletion continue on partial failure (databases are independent)
- **Production protection**: Production environments cannot be deleted programmatically - this is hard-coded
- **Soft delete**: Deleted databases are soft-deleted in `master_config.database_configs` (`is_active = FALSE`) for audit trail

## See Also

- `TFP-Algo/algo/00_docs/database_environment_management.md` - User documentation
- Implementation plan for detailed architecture and setup instructions
