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

### db_utils.py

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

### db_management.py

CLI and library functions for environment lifecycle: single-database create, full-env clone, diff/sync, and delete.

**CLI entry point:** `python -m dbase.database.db_management` (preferred over invoking `db_management.py` by path).

| Subcommand | Purpose |
|------------|---------|
| `create-db` | One empty MySQL database + `master_config.database_configs` row |
| `create` | Clone **all** databases from `--source-env` (schema-only or with-data) |
| `delete` | Drop env databases (protected envs blocked) |
| `list` | List environments in registry (excludes protected by default) |
| `diff` | Compare source vs target (missing DBs/tables) |
| `sync` | Apply missing DBs/tables (dry run unless `--apply`) |

**Makefile:** From repo root, `make help` (includes `dbase/database/Makefile` via root `Makefile`). Or `make -C dbase/database help`.

#### Key Functions

**`create_database_for_environment(base_name, environment, branch_name, dry_run)`**

- Create a single empty MySQL database and register it in `master_config.database_configs`
- Physical name: `base_name` for `prod`, else `{base_name}_{environment}`
- Does not clone schema (use `create_test_environment` for full env clones)
- **Caution:** `create-db` with `ENV=prod` is allowed (unlike `delete`, which blocks prod). Creates an empty production-named schema — use only when intentional.

**`create_test_environment(environment, branch_name, source_environment, exclude_databases, schema_only)`**

- Main function for creating environments by cloning a **required** source environment
- Clones all databases from `source_environment` (no default; CLI `--source-env` is required)
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
- Protected environments cannot be deleted programmatically (see `DB_PROTECTED_ENVIRONMENTS` below)
- Returns nested dict: `{environment: {base_name: database_name}}`

**`get_protected_environments()`**

- Returns environment names blocked from programmatic delete and excluded from `list_environments()` when `exclude_prod=True`
- Source: `DB_PROTECTED_ENVIRONMENTS` env var (empty when unset; operators must set explicitly)

**`list_environments(exclude_prod)`**

- Lists all unique environments from `master_config.database_configs`
- When `exclude_prod=True` (default), excludes all protected environments from `get_protected_environments()`
- Returns sorted list of environment names

#### Environment diff and sync

**`diff_environments(source_environment, target_environment)`**

- Compares two environments: what databases and tables does the target lack relative to the source?
- Returns `EnvironmentDiff` with `missing_databases` (base_name -> source DB name) and `table_differences` (per-base tables missing in target).

**`EnvironmentDiff`**

- Dataclass: `source_environment`, `target_environment`, `missing_databases: dict[str, str]`, `table_differences: dict[str, dict]`.

**`sync_environment_from_source(source_environment, target_environment, branch_name, ...)`**

- One-shot: create missing databases and add missing tables in target from source.
- **Dry run by default** (`apply=False`). Set `apply=True` to perform changes.
- Returns created_databases, failed_databases, synced_tables, failed_tables (partial success on failures).

**`create_missing_databases_from_environment(...)`** / **`sync_missing_tables_from_environment(...)`**

- Lower-level helpers; both default to dry run (`apply=False`).

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

### Creating a single empty database

```python
from dbase.database import create_database_for_environment

# Empty DB + registry row (no schema clone)
physical = create_database_for_environment(
    base_name="portfolio_config",
    environment="scratch",
    branch_name="feature-branch",
)
# physical == "portfolio_config_scratch" (prod would return "portfolio_config")

# Preview only
create_database_for_environment(
    base_name="portfolio_data",
    environment="scratch",
    branch_name="feature-branch",
    dry_run=True,
)
```

### Creating a full environment (clone all databases)

```python
from dbase.database import create_test_environment

# Clone every database from source env (--source-env required on CLI)
created = create_test_environment(
    environment="mean-reversion",
    branch_name="feature-branch",
    source_environment="long_bbands_v2",
    schema_only=True,  # skeleton only (--no-data); bot tables empty until manual seed
)

# Returns: {'portfolio_data': 'portfolio_data_mean-reversion', ...}
# Note: clone path uses f"{base_name}_{environment}" for all envs including non-prod naming;
# create_database_for_environment uses base_name only when environment == "prod".
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

# Protected environments raise ValueError when listed in DB_PROTECTED_ENVIRONMENTS
# export DB_PROTECTED_ENVIRONMENTS='["prod"]'
# delete_environment(['prod'])  # ERROR: Cannot delete protected environment(s)
```

### Protected environments (`DB_PROTECTED_ENVIRONMENTS`)

Delete and `list` (default) treat certain registry environment names as protected. Configure with:

```bash
# JSON array
export DB_PROTECTED_ENVIRONMENTS='["prod","long_bbands_v2"]'

# Comma-separated (whitespace trimmed)
export DB_PROTECTED_ENVIRONMENTS='prod,long_bbands_v2'
```

When unset or empty after parse, no environments are protected. Each entry must pass the same validation as other database/environment names.

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

### Physical database naming

| Tool / API | `environment == "prod"` | Other environments |
|------------|-------------------------|---------------------|
| `create_database_for_environment` / `create-db` | `portfolio_data` | `portfolio_data_{env}` |
| `get_database_name()` (runtime) | base name | from `master_config` |
| `create_test_environment` (clone) | `{base}_{env}` | `{base}_{env}` |

Use **`create-db`** when adding one registry-backed database. Use **`create`** when bootstrapping a full environment from a template source.

### Schema cloning process (`create` / `create_test_environment`)

1. Query `master_config.database_configs` for source environment databases
2. For each database:
   - Check for name conflicts
   - Clone schema (and optionally data) using mysqldump
   - Register in `master_config.database_configs`
3. After schema-only create, bot config tables are **empty** — seed manually (see TFP-Algo `seed_bot_config`) if needed
4. Return mapping of created databases

### Single-database create process (`create-db`)

1. Validate `base_name` and `environment`; reject protected names (`master_config`, system schemas)
2. Resolve physical name (see table above)
3. Fail if MySQL schema already exists or active registry row conflicts
4. `CREATE DATABASE` (empty, utf8)
5. Insert or reactivate row in `master_config.database_configs`

### Environment Deletion Process

1. Validate all environment names; reject any in `DB_PROTECTED_ENVIRONMENTS`
2. Collect all databases for each environment
3. Display databases grouped by environment (if confirmation required)
4. Prompt for user confirmation (`y/n`)
5. For each database:
   - Delete MySQL database using `DROP DATABASE IF EXISTS`
   - Soft delete from `master_config.database_configs` (set `is_active = FALSE`)
6. Return mapping of deleted databases

## CLI and Makefile

Run from **FinanceDatabase repo root** with `PYTHONPATH=.` (Makefiles set this automatically).

```bash
# Help and tests (unit tests: test_db_management_create*.py)
make help
make test
# or: make -C dbase/database help
```

### Makefile targets (`dbase/database/Makefile`)

Variables default to `none`; required ones are validated with a clear error.

| Target | Required variables | Notes |
|--------|-------------------|--------|
| `create-db` | `ENV`, `NAME` | Empty DB + registry; optional `BRANCH`, `ARGS=--dry-run` |
| `create-env` | `ENV`, `BRANCH`, `SOURCE_ENV` | Schema-only skeleton; optional `EXCLUDE=portfolio_data` |
| `create-env-with-data` | same as `create-env` | Copies all table data from source |
| `delete` | `ENV` | Full environment name; `CONFIRM=1` skips prompt |
| `diff` | `SOURCE_ENV`, `TARGET_ENV` | Read-only comparison |
| `sync` | `SOURCE_ENV`, `TARGET_ENV` | Dry run |
| `sync-apply` | `SOURCE_ENV`, `TARGET_ENV` | `WITH_DATA=1` copies data; optional `BRANCH` |
| `list` | — | Registry environments (excludes protected; see `DB_PROTECTED_ENVIRONMENTS`) |
| `test` | — | Unit tests under `tests/test_db_management_*.py` |

**`ENV` naming:** For `create-env` / `create-db`, use the **short** environment key (e.g. `mean-reversion`, `long_bbands_v2`). For `delete`, use the **full** registry environment string (e.g. `test-mean-reversion`) as stored in `database_configs.environment`.

```bash
# Single empty database
make -C dbase/database create-db ENV=scratch NAME=portfolio_config BRANCH=feature-x
make -C dbase/database create-db ENV=scratch NAME=portfolio_data ARGS=--dry-run

# Full environment from template (no default --source-env)
make -C dbase/database create-env ENV=mean-reversion BRANCH=feature-x SOURCE_ENV=long_bbands_v2
make -C dbase/database create-env-with-data ENV=mean-reversion BRANCH=feature-x SOURCE_ENV=long_bbands_v2

make -C dbase/database list
make -C dbase/database diff SOURCE_ENV=long_bbands_v2 TARGET_ENV=scratch
make -C dbase/database sync SOURCE_ENV=long_bbands_v2 TARGET_ENV=scratch
make -C dbase/database sync-apply SOURCE_ENV=long_bbands_v2 TARGET_ENV=scratch WITH_DATA=1
make -C dbase/database delete ENV=test-mean-reversion CONFIRM=1
```

### CLI (`python -m dbase.database.db_management`)

#### `create-db` — one empty database

```bash
python -m dbase.database.db_management create-db \
    --env scratch \
    --name portfolio_config \
    --branch feature-x

python -m dbase.database.db_management create-db \
    --env scratch \
    --name portfolio_data \
    --dry-run
```

`--branch` optional (defaults to current git branch via `ALGO_DIR` or `.`).

#### `create` — clone full environment

`--source-env` is **required** (no default).

```bash
# Schema-only: all tables empty (including discord bot config)
python -m dbase.database.db_management create \
    --env mean-reversion \
    --branch feature-branch \
    --source-env long_bbands_v2 \
    --schema-only

# With data: rows copied from source for every table
python -m dbase.database.db_management create \
    --env mean-reversion \
    --branch feature-branch \
    --source-env long_bbands_v2 \
    --with-data
```

`create` does **not** call `seed_bot_config`. After schema-only create, seed bot config manually if needed (TFP-Algo: `make seed-bot-config ENV=... SOURCE_ENV=...`).

#### `list`, `delete`, `diff`, `sync`

```bash
python -m dbase.database.db_management list

python -m dbase.database.db_management delete --delete-env test-mean-reversion
python -m dbase.database.db_management delete --delete-env test-mean-reversion --confirm

python -m dbase.database.db_management diff \
    --source-env long_bbands_v2 \
    --target-env scratch

python -m dbase.database.db_management sync \
    --source-env long_bbands_v2 \
    --target-env scratch

python -m dbase.database.db_management sync \
    --source-env long_bbands_v2 \
    --target-env scratch \
    --branch feature-branch \
    --with-data \
    --apply
```

Sync is **dry run by default**; pass `--apply` to change MySQL.

### Safety

| Action | Prod behavior |
|--------|----------------|
| `delete` | **Blocked** for envs in `DB_PROTECTED_ENVIRONMENTS` |
| `create-db` | **Allowed** — creates empty DB with production base name; use only when intentional |
| `create` | Allowed with `--source-env`; clones all registered DBs for that source |

- Delete confirmation is required by default (`--confirm` or `CONFIRM=1` to skip).
- Schema-only `create` leaves bot tables empty by design (skeleton contract).

## Discord bot config (portfolio_config)

Discord bot routing (channels, persona, client allowlist) lives in **three normalized tables** inside each environment's physical `portfolio_config_*` database. TFP-Algo resolves them at runtime after `set_environment_context()`; see the audit bundle for architecture and operator flows.

### Canonical source environment

**`long_bbands_v2`** (`portfolio_config_long_bbands_v2`) — reference schema + Emefiele seed rows. Use as `--source-env` for every `db_management create` / sync that should inherit bot table structure (and optionally data).

Legacy `prod`, `test`, and `long_bbands` are **not** retrofit targets; new prod is created from this template, then persona is set via manual seed (Tinubu yaml seed).

### Reference DDL (plan §4.1)

No versioned migration files in this repo — schema propagates via `create` / `sync` (`CREATE TABLE … LIKE` or full clone). Apply manually only when bootstrapping the canonical env; otherwise clone from `long_bbands_v2`.

```sql
CREATE TABLE discord_channels (
  id            INT NOT NULL AUTO_INCREMENT,
  channel_type  VARCHAR(32) NOT NULL,
  channel_id    BIGINT UNSIGNED NOT NULL,
  channel_name  VARCHAR(64) NOT NULL,
  guild_id      BIGINT UNSIGNED NULL,
  enabled       TINYINT(1) NOT NULL DEFAULT 1,
  notes         VARCHAR(255) NULL,
  updated_at    TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  updated_by    VARCHAR(64) NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uq_discord_channels_type (channel_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE bot_clients (
  id               INT NOT NULL AUTO_INCREMENT,
  discord_username VARCHAR(64) NOT NULL,
  enabled          TINYINT(1) NOT NULL DEFAULT 1,
  notes            VARCHAR(255) NULL,
  updated_at       TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_bot_clients_username (discord_username)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE bot_identity (
  id             INT NOT NULL AUTO_INCREMENT,
  bot_name       VARCHAR(32) NOT NULL,
  token_env_key  VARCHAR(64) NOT NULL,
  notes          VARCHAR(255) NULL,
  updated_at     TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

### `create` and bot rows

`--source-env` is **required** on `db_management create` (no default).

| Mode | Bot tables | Rows |
|------|------------|------|
| `--schema-only` (default) | Cloned structure | **0** — skeleton; operator seeds manually |
| `--with-data` | Cloned structure | Copied from named source (persona = source `bot_identity`) |

`create` does **not** invoke `seed_bot_config`. After schema-only create, post-create logs may suggest TFP-Algo seed/verify commands.

```bash
python -m dbase.database.db_management create \
  --env <new> --branch <branch> \
  --source-env long_bbands_v2 --schema-only
```

### Manual seed after skeleton (TFP-Algo)

From **TFP-Algo repo root**:

```bash
# Copy Emefiele layout from canonical env
make seed-bot-config ENV=<new> SOURCE_ENV=long_bbands_v2

# New prod Tinubu from algo/bot/config.yaml
make seed-bot-config-yaml ENV=prod PERSONA=tinubu
```

Use `FORCE=1` / `--force` when target already has bot rows (non-interactive overwrite).

### Verify

**CLI** (TFP-Algo):

```bash
make verify-bot-config ENV=<env>
make verify-bot-config ENV=<env> COMPARE_YAML=1   # channel IDs vs config.yaml
make verify-bot-config-verbose ENV=<env>
```

Exit `0` = no errors (warnings allowed for optional channel types).

**MCP / SQL** (replace DB name for target env):

```sql
-- Channel counts and IDs
SELECT channel_type, channel_name, channel_id, enabled
FROM portfolio_config_<env>.discord_channels
ORDER BY channel_type;

-- Exactly one identity row
SELECT id, bot_name, token_env_key FROM portfolio_config_<env>.bot_identity;

-- Enabled clients
SELECT discord_username, enabled
FROM portfolio_config_<env>.bot_clients
WHERE enabled = 1;
```

Pilot canonical example: `portfolio_config_long_bbands_v2` — expect 4 enabled channels, 1 identity (`emefiele` / `EMEFIELE_TOKEN`), 2 clients.

### Rollback (runtime)

Set **`BOT_CONFIG_SOURCE=yaml`** on the bot process and restart. Default runtime path is DB-only (Phase 4). Tokens stay in `.env`; only `token_env_key` names are stored in `bot_identity`.

### TFP-Algo audit bundle

| Doc | Path |
|-----|------|
| Audit | [../../TFP-Algo/audits/discord-config-db-migration-option2/audit.md](../../TFP-Algo/audits/discord-config-db-migration-option2/audit.md) |
| Fix checklist | [../../TFP-Algo/audits/discord-config-db-migration-option2/fix-checklist.md](../../TFP-Algo/audits/discord-config-db-migration-option2/fix-checklist.md) |
| Fix flow | [../../TFP-Algo/audits/discord-config-db-migration-option2/fix-flow.md](../../TFP-Algo/audits/discord-config-db-migration-option2/fix-flow.md) |
| Operator README | [../../TFP-Algo/audits/discord-config-db-migration-option2/README.md](../../TFP-Algo/audits/discord-config-db-migration-option2/README.md) |
| Strategic plan | [../../TFP-Algo/audits/discord-config-db-migration-option2-plan.md](../../TFP-Algo/audits/discord-config-db-migration-option2-plan.md) |
| Post-completion | [../../TFP-Algo/audits/discord-config-db-migration-option2/post-completion.md](../../TFP-Algo/audits/discord-config-db-migration-option2/post-completion.md) |

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

- **Production databases**: Resolved at runtime via `get_database_name()` as base names (e.g. `portfolio_data`)
- **Non-prod physical names**: Typically `{base_name}_{environment}` when created via `create-db` or clone
- **master_config**: Never suffixed; cannot be created via `create-db`
- **Caching**: Database name resolution is cached per environment/base_name combination
- **Partial failures**: Schema cloning and deletion continue on partial failure (databases are independent)
- **Soft delete**: Deleted databases are soft-deleted in `master_config.database_configs` (`is_active = FALSE`) for audit trail
- **TFP-Algo bot config**: After schema-only env create, use `seed_bot_config` / `verify_bot_config` in TFP-Algo (see `TFP-Algo/Makefile`)

## See Also

- `TFP-Algo/algo/00_docs/database_environment_management.md` - User documentation
- Implementation plan for detailed architecture and setup instructions
