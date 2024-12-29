# SQLHelpers.py

## Overview

`SQLHelpers.py` is a Python module that provides utility functions for interacting with a SQL database using SQLAlchemy. The module includes functions for creating database engines, executing queries, and updating records in a specified table.

## Functions

### `create_engine_short(db)`

Creates a SQLAlchemy engine for connecting to a specified database.

**Parameters:**
- `db`: The name of the database to connect to.

**Returns:**
- A SQLAlchemy engine object.

### `store_SQL_data(db, table_name, data)`

Stores a pandas DataFrame into a specified table in the database.

**Parameters:**
- `db`: Database connection string.
- `table_name`: The name of the table to store the data in.
- `data`: The pandas DataFrame to store.

### `query_database(db, sq_table_name, query)`

Executes a SQL query on a specified database and returns the result as a pandas DataFrame.

**Parameters:**
- `db`: The name of the database to query.
- `sq_table_name`: The name of the table to query.
- `query`: The SQL query string to execute.

**Returns:**
- A pandas DataFrame containing the query results.

### `close_SQL_connection(connection, cursor=None)`

Closes an open MySQL connection and optionally a cursor.

**Parameters:**
- `connection`: The MySQL connection object to close.
- `cursor`: (Optional) The cursor object to close.

### `update_table(db, table_name, update_values, condition)`

Updates records in a specified table based on given conditions.

**Parameters:**
- `db`: Database connection string.
- `table_name`: The name of the table to update.
- `update_values`: Dictionary of column-value pairs to update.
- `condition`: Dictionary of column-value pairs to use as conditions for the update.

### `execute_query(db, table_name, query, params=None)`

Executes a query on a specified table in the database.

**Parameters:**
- `db`: Database connection string.
- `table_name`: The name of the table to execute the query on.
- `query`: The SQL query to execute.
- `params`: Dictionary of parameters for the query (optional).

## Usage

1. **Create Engine:**
   ```python
   engine = create_engine_short('your_database_name')



**Note: this README.md is AI generated**