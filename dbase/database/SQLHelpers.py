import logging
import sys, os
from dotenv import load_dotenv  
load_dotenv()
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, Date, Float, Boolean, Enum, Time, DateTime, TIMESTAMP, PrimaryKeyConstraint
from sqlalchemy import create_engine, text
import pymysql
import pandas as pd
from mysql.connector import Error
import sys
import pandas as pd
from datetime import datetime
from trade.helpers.helper import setup_logger, _ipython_shutdown
from trade import register_signal
import mysql.connector
import os
from functools import lru_cache
from dotenv import load_dotenv
import atexit
import signal
load_dotenv()
sql_pw = (os.environ.get('MYSQL_PASSWORD'))
sql_host = (os.environ.get('MYSQL_HOST'))
sql_port = (os.environ.get('MYSQL_PORT'))
sql_user = os.environ.get('MYSQL_USER')

sys.path.append(os.environ['DBASE_DIR'])

logger = setup_logger('dbase.database.SQLHelpers')

"""
This module is responsible for organizing all functions necessary for accessing/retrieving data from SQL Database
"""


logger = setup_logger('SQLHelpers.py')  # Using a module-specific logger
_PROCESS_ENGINE_CACHE = {}
_PYMYSQL_CONNECTION_CACHE = {}
dbs = ['securities_master', 'vol_surface']


def create_engine_short(db):
    """
    Create a SQLAlchemy engine for the given database.
    """
    return create_engine(f"mysql+mysqlconnector://{sql_user}:{sql_pw}@{sql_host}/{db}")

def create_pymysql_connection(db):
    """
    Create a pymysql connection for the given database.
    """
    connection = pymysql.connect(host=sql_host,
                                 user=sql_user,
                                 password=sql_pw,
                                 database=db,
                                 cursorclass=pymysql.cursors.DictCursor)
    return connection

def get_engine(db_name):
    """
    Get a SQLAlchemy engine for the given database name and process ID. This saves to a cache to avoid creating multiple engines for the same database in the same process.
    """
    pid = os.getpid()
    key = (pid, db_name)

    if key not in _PROCESS_ENGINE_CACHE:
        print(f"[get_engine] Creating engine for DB: {db_name}, PID: {pid}")
        _PROCESS_ENGINE_CACHE[key] = create_engine_short(db_name)

    return _PROCESS_ENGINE_CACHE[key]

def get_pymysql_connection(db_name):
    """
    Get a pymysql connection for the given database name and process ID. This saves to a cache to avoid creating multiple connections for the same database in the same process.
    """
    pid = os.getpid()
    key = (pid, db_name)

    if key not in _PYMYSQL_CONNECTION_CACHE:
        print(f"[get_pymysql_connection] Creating connection for DB: {db_name}, PID: {pid}")
        _PYMYSQL_CONNECTION_CACHE[key] = create_pymysql_connection(db_name)

    return _PYMYSQL_CONNECTION_CACHE[key]


def _dispose_all_engines(*args, **kwargs):
    """
    Dispose all SQLAlchemy engines and close all pymysql connections.
    """
    for pid, engine in _PROCESS_ENGINE_CACHE.items():
        try:
            engine.dispose()
            with open(f'{os.environ["DBASE_DIR"]}/logs/atexit.log', 'a') as f:
                f.write(f"Engine disposed: {str(pid)} on {datetime.now()}\n")
        except Exception as e:
            with open(f'{os.environ["DBASE_DIR"]}/logs/atexit.log', 'a') as f:
                f.write(f"Error Disposing Engine: {str(pid)}, {e} on {datetime.now()}\n")
            pass
    for conn in _PYMYSQL_CONNECTION_CACHE.values():
        try:
            conn.close()
            with open(f'{os.environ["DBASE_DIR"]}/logs/atexit.log', 'a') as f:
                f.write(f"Connection closed: {str(pid)} on {datetime.now()}\n")
        except Exception as e:
            with open(f'{os.environ["DBASE_DIR"]}/logs/atexit.log', 'a') as f:
                f.write(f"Error Closing Connection: {str(pid)}, {e} on {datetime.now()}\n")
            pass

def signal_exit_handler(signum, frame):
    """
    Signal handler for graceful shutdown. It disposes all engines and closes all connections before exiting.
    """

    _dispose_all_engines()
    sys.exit(0)  # Exit the program gracefully

## .py exit kill with atexit
atexit.register(_dispose_all_engines)

##.py exit kill with signal
# signal.signal(signal.SIGTERM, signal_exit_handler)
register_signal(signal.SIGTERM, _dispose_all_engines)

def store_SQL_data(db, sql_table_name, data, if_exists='append'):
    """
    Store data in a SQL table. If the table does not exist, it will be created.
    """
    engine = get_engine(db)
    data.to_sql(sql_table_name, engine, if_exists=if_exists, index=False)
    print('Data successfully saved', end = '\r')


def drop_SQL_Table_Duplicates(db, sql_table_name):
    """
    Drop duplicates from a SQL table and replace the original table with the non-duplicated data.
    """
    
    
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
    logger.info('Duplicates succesfully dropped', end = '\r')


def query_database(db, tbl_name, query):
    """
    Query the database using SQLAlchemy and return the result as a pandas DataFrame.
    """

    engine = get_engine(db)
    return pd.read_sql(query, engine)

def _query_database_testing(db,tbl_name, query):
    """
    Query the database using pymysql and return the result as a pandas DataFrame.
    """

    conn = get_pymysql_connection(db)
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()                   # this is a list of dicts
        return pd.DataFrame.from_records(rows)     # “fast path” into a DataFrame
    except pymysql.Error as e:
        logger.error(f"Error executing query: {e}")
        raise e


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
        logger.info("Successfully connected to the database", end = '\r')
    except mysql.connector.Error as err:
        logger.info(err.errno)
        logger.info(err.msg)

    return connection


def close_SQL_connection(connection, cursor=None):
    if connection.is_connected():
        cursor.close() if cursor else None
        connection.close()
        logger.info("MySQL connection is closed", end = '\r')


def create_SQL_database(connection, db_name):
    cursor = connection.cursor()
    try:
        cursor.execute(
            "CREATE DATABASE {} DEFAULT CHARACTER SET 'utf8'".format(db_name))
    except mysql.connector.Error as err:
        print("Failed creating database: {}".format(err))
        exit(1)
    connection.commit()
    logger.info("Database created successfully", end = '\r')
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
        logger.info(
            f"Table '{table_name}' has been created with columns: {[col.name for col in column_definitions]}", end = '\r')
    except SQLAlchemyError as e:
        logger.info(f"An error occurred: {e}")


def store_SQL_data_Insert_Ignore(db, sql_table_name, data, _raise=False):
    """
    Store data in a SQL table using INSERT IGNORE. If the table does not exist, it will be created.
    """
    engine =create_engine_short(db)
    logger.info(f"Size to be inserted: {len(data)}")
    with engine.begin() as connection:
        connection.execute(text(f"""
            CREATE TEMPORARY TABLE temp LIKE {sql_table_name};
        """))
        try:
            data.to_sql('temp', con=connection, if_exists='append',
                        index=False, chunksize=1000)
            print("Data inserted into temporary table.", end = '\r')
        except Exception as e:
            logger.error(f"Error during insertion into temp: {e}")
            if _raise:
                raise e

        try:
            result = connection.execute(text(f"""
                INSERT IGNORE INTO {sql_table_name}
                SELECT * FROM temp;
            """))
            print(f"Rows inserted into {sql_table_name}: {result.rowcount}", end = '\r')
        except Exception as e:
            logger.error(f"Error during INSERT IGNORE: {e}")
            if _raise:
                raise e

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

    engine = get_engine(db)
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

    engine = get_engine(db)

    # Prepare the query
    query = text(query)

    # Execute the query
    with engine.begin() as conn:
        conn.execute(query, params or {})
        logger.info("Query executed successfully.", end = '\r')



class DatabaseAdapter:

    def __init__(self):
        pass

    def save_to_database(self, data, db, table_name, filter_data=True, _raise = False):
        data = self.__filter_data(data) if filter_data else data
        store_SQL_data_Insert_Ignore(db, table_name, data, _raise=_raise)

    def query_database(self, db, table_name, query):
        data = query_database(db, table_name, query)
        return data

    def __filter_data(self, data):

        ## To-doL Add a warning log here for dropping second duplicate columns
        data.columns = [col.lower() for col in data.columns]
        
        ## Print Data Info
        na_rows = data.isna().any(axis=1).sum()
        na_cols = data.isna().any(axis = 0)
        
        logger.info("Columns with NaN") if na_rows else None
        logger.info(na_cols[na_cols==True]) if na_rows else None
        logger.error(f"Rows with at least one NA: {na_rows}") if na_rows else None

        dup_rows = data.duplicated().sum()
        logger.info(f"Fully duplicated rows: {dup_rows}") if dup_rows else None



        ## Filter duplicate data
        data = data.drop_duplicates()
        data = data.dropna()

        ## Ensure no duplicate columns
        data.columns = data.columns.str.lower()
        occurence = dict(zip(data.columns, [0 for _ in range(len(data.columns))]))
        ## Tag only second columns of duplicated
        for j, i in enumerate(data.columns):
            occurence[i] += 1
            if occurence[i] > 1:
                data.columns.values[j] = f"{i}_dup"
        dup_names = [x for x in data.columns if 'dup' in x]
        data.drop(columns = dup_names, inplace = True)

        return data
