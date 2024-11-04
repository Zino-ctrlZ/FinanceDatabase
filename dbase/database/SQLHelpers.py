# from trade.assets.Calculate import Calculate
# from trade.assets.Option import Option
import numpy as np
import logging
import sys, os
from dotenv import load_dotenv  
load_dotenv()
sys.path.extend(
    [ os.environ.get('DBASE_DIR'),  os.environ.get('WORK_DIR')])
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, Date, Float, Boolean, Enum, Time, DateTime, TIMESTAMP, PrimaryKeyConstraint
from sqlalchemy import create_engine, text
from mysql.connector import Error
import sys
import pandas as pd
from datetime import datetime
from trade.helpers.helper import setup_logger
import mysql.connector
import os
from dotenv import load_dotenv
load_dotenv()
sql_pw = (os.environ.get('MYSQL_PASSWORD'))
sql_host = (os.environ.get('MYSQL_HOST'))
sql_port = (os.environ.get('MYSQL_PORT'))
sql_user = os.environ.get('MYSQL_USER')
sql_user
sys.path.append(
    os.environ.get('WORK_DIR'))

"""
This module is responsible for organizing all functions necessary for accessing/retrieving data from SQL Database
"""

# Inside the imported module
# logger = logging.getLogger(__name__)  # Using a module-specific logger
# logger.error('An error occurred in the module')
# logger.propagate = True  # Ensure it propagates to the root logger

logger = setup_logger('SQLHelpers.py')  # Using a module-specific logger

def create_engine_short(db):
    return create_engine(f"mysql+mysqlconnector://{sql_user}:{sql_pw}@{sql_host}/{db}")


def store_SQL_data(db, sql_table_name, data, if_exists='append'):
    # ADD INITIAL DATA TO DATABASE
    engine = create_engine_short(db)
    data.to_sql(sql_table_name, engine, if_exists=if_exists, index=False)
    print('Data successfully saved')


def drop_SQL_Table_Duplicates(db, sql_table_name):
    # RE-QUERY WHOLE DATA
    df = query_database(db, sql_table_name,
                        f"SELECT * FROM {db}.{sql_table_name}")

    # DROP DUPLICATES OF WHOLE DATA
    df = df.drop_duplicates()
    range_ = list(range(0, len(df), 10000),)
    range_.append(len(df))

    # Implementing a batch save
    for i, i2 in enumerate(list(range(0, len(df), 10000)), start=1):
        start = i2
        end = i
        if start != range_[-1]:
            if i != 1:
                print(i2+1, range_[i])
                start, end = i2+1, range_[i]
            else:
                print(i2, range_[i])
                start, end = i2, range_[i]
        else:
            print(i2, range_[-1])
            start, end = i2, range_[-1]
        use_df = df.iloc[start:end+1, :]
        if i == 1:
            store_SQL_data(db, sql_table_name, use_df, if_exists='replace')
        else:
            store_SQL_data(db, sql_table_name, use_df, if_exists='append')
    # REPLACE INITIAL TABLE WITH NON DUPLICATED TABLE
    print('Duplicates succesfully dropped')


def query_database(db, sq_table_name, query):
    engine = create_engine_short(db)
    return pd.read_sql(query, engine)


def create_SQL_connection():

    try:
        connection = mysql.connector.connect(
            # The IP address or domain name of your MySQL server (e.g., '192.168.1.100' or 'yourdomain.com')
            host=sql_host,
            # MySQL port number (typically 3306)
            port=sql_port,
            database='securities_master',     # The name of the database you want to connect to
            user=sql_user,         # Your MySQL username
            password=sql_pw     # Your MySQL password
        )
        print("Successfully connected to the database")
    except mysql.connector.Error as err:
        print(err.errno)
        print(err.msg)

    return connection


def close_SQL_connection(connection, cursor=None):
    if connection.is_connected():
        cursor.close() if cursor else None
        connection.close()
        print("MySQL connection is closed")


def create_SQL_database(connection, db_name):
    cursor = connection.cursor()
    try:
        cursor.execute(
            "CREATE DATABASE {} DEFAULT CHARACTER SET 'utf8'".format(db_name))
    except mysql.connector.Error as err:
        print("Failed creating database: {}".format(err))
        exit(1)
    connection.commit()
    print("Database created successfully")
    close_SQL_connection(connection, cursor)


def create_table_from_schema(engine, table_schema):
    """
    Creates a table in the database based on the provided schema dictionary.

    Parameters:
    - engine (sqlalchemy.engine.base.Engine): SQLAlchemy engine object connected to the database.
    - table_schema (dict): Dictionary defining the table schema. The dictionary should have the following structure:
      {
          'table_name': 'your_table_name',
          'columns': [
              {'name': 'column1', 'type': 'Integer', 'primary_key': True, 'nullable': False},
              {'name': 'column2', 'type': 'String', 'length': 50, 'nullable': False},
              {'name': 'column3', 'type': 'Date'},
              {'name': 'column4', 'type': 'Enum', 'values': ['value1', 'value2']},
              {'name': 'column5', 'type': 'Time'},
              {'name': 'column6', 'type': 'DateTime'},
              {'name': 'column7', 'type': 'Timestamp'}
          ]
      }

    Example:
    >>> engine = create_engine('mysql+mysqlconnector://user:password@host:port/database')
    >>> schema = {
    >>>     'table_name': 'users',
    >>>     'columns': [
    >>>         {'name': 'id', 'type': 'Integer', 'primary_key': True, 'nullable': False},
    >>>         {'name': 'name', 'type': 'String', 'length': 100, 'nullable': False},
    >>>         {'name': 'signup_date', 'type': 'Date'},
    >>>         {'name': 'status', 'type': 'Enum', 'values': ['active', 'inactive']},
    >>>         {'name': 'last_login_time', 'type': 'Time'},
    >>>         {'name': 'created_at', 'type': 'DateTime'},
    >>>         {'name': 'updated_at', 'type': 'Timestamp'}
    >>>     ]
    >>> }
    >>> create_table_from_schema(engine, schema)
    """

    # Initialize metadata object
    metadata = MetaData()

    # Extract table name and columns from schema
    table_name = table_schema.get('table_name')
    columns = table_schema.get('columns', [])

    # Define columns for the table
    column_definitions = []
    primary_keys = []
    for col in columns:
        col_name = col.get('name')
        col_type = col.get('type')
        col_length = col.get('length', None)
        col_values = col.get('values', None)
        is_primary_key = col.get('primary_key', False)
        is_nullable = col.get('nullable', True)

        # Define column type
        if col_type == 'Integer':
            column_type = Integer
        elif col_type == 'String':
            if col_length is None:
                raise ValueError(
                    f"Length must be specified for String type column: {col_name}")
            column_type = String(col_length)
        elif col_type == 'Date':
            column_type = Date
        elif col_type == 'Float':
            column_type = Float
        elif col_type == 'Boolean':
            column_type = Boolean
        elif col_type == 'Enum':
            if col_values is None or not isinstance(col_values, list):
                raise ValueError(
                    f"Values must be specified for Enum type column: {col_name}")
            column_type = Enum(*col_values)
        elif col_type == 'Time':
            column_type = Time
        elif col_type == 'DateTime':
            column_type = DateTime
        elif col_type == 'Timestamp':
            column_type = TIMESTAMP
        else:
            raise ValueError(f"Unsupported column type: {col_type}")

        # Add column definition
        column_definition = Column(col_name, column_type, nullable=is_nullable)
        if is_primary_key:
            primary_keys.append(col_name)
        column_definitions.append(column_definition)

    # Define the table
    table = Table(table_name, metadata, *column_definitions)

    # Add primary key constraint if any
    if primary_keys:
        table.append_constraint(PrimaryKeyConstraint(*primary_keys))

    # Create the table in the database
    try:
        metadata.create_all(engine)
        print(
            f"Table '{table_name}' has been created with columns: {[col.name for col in column_definitions]}")
    except SQLAlchemyError as e:
        print(f"An error occurred: {e}")


def store_SQL_data_Insert_Ignore(db, sql_table_name, data):
    engine = create_engine_short(db)

    with engine.begin() as connection:
        connection.execute(text(f"""
            CREATE TEMPORARY TABLE temp LIKE {sql_table_name};
        """))

        try:
            data.to_sql('temp', con=connection, if_exists='append',
                        index=False, chunksize=1000)
            print("Data inserted into temporary table.")
        except Exception as e:
            print(f"Error during insertion into temp: {e}")

        try:
            result = connection.execute(text(f"""
                INSERT IGNORE INTO {sql_table_name}
                SELECT * FROM temp;
            """))
            print(f"Rows inserted into {sql_table_name}: {result.rowcount}")
        except Exception as e:
            print(f"Error during INSERT IGNORE: {e}")

        connection.execute(text("DROP TABLE temp;"))


def dynamic_batch_update(db, table_name, update_values, condition):
    """
    Update multiple columns in a table dynamically.

    Parameters:
    - engine: SQLAlchemy engine object for database connection.
    - table_name: The name of the table to update.
    - update_values: Dictionary of columns and their new values to update.
    - condition: Dictionary of conditions for the WHERE clause.
    """

    engine = create_engine_short(db)
    # Create the SET clause
    set_clause = ", ".join([f"{col} = :{col}" for col in update_values.keys()])

    # Create the WHERE clause
    where_clause = " AND ".join(
        [f"{col} = :cond_{col}" for col in condition.keys()])

    # Prepare the query
    query = text(f"""
        UPDATE {table_name}
        SET {set_clause}
        WHERE {where_clause}
                    """)

    # Combine parameters for execution
    params = {**update_values, **
              {f'cond_{col}': val for col, val in condition.items()}}

    with engine.begin() as conn:
        conn.execute(query, params)

def execute_query(db, table_name, query, params=None):
    """
    Execute a query on a specified table in the database.

    Parameters:
    - db: Database connection string.
    - table_name: The name of the table to execute the query on.
    - query: The SQL query to execute.
    - params: Dictionary of parameters for the query (optional).
    """

    engine = create_engine_short(db)

    # Prepare the query
    query = text(query)

    # Execute the query
    with engine.begin() as conn:
        conn.execute(query, params or {})
        print("Query executed successfully.")

