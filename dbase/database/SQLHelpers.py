from dotenv import load_dotenv
load_dotenv()
import os
import sys
sql_pw = (os.environ.get('MYSQL_PASSWORD'))
sql_host = (os.environ.get('MYSQL_HOST'))
sql_port = (os.environ.get('MYSQL_PORT'))
sql_user = os.environ.get('MYSQL_USER')
sql_user
sys.path.append(
    os.environ.get('WORK_DIR'))
from trade.assets.Option import Option
from trade.assets.Calculate import Calculate
import pandas as pd
from datetime import datetime
import mysql.connector
from mysql.connector import Error
from sqlalchemy import create_engine
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, Date, Float, Boolean, Enum, Time, DateTime, TIMESTAMP, PrimaryKeyConstraint
from sqlalchemy.exc import SQLAlchemyError



def create_engine_short(db):
    return create_engine(f"mysql+mysqlconnector://{sql_user}:{sql_pw}@{sql_host}/{db}")


def store_SQL_data(db, sql_table_name, data, if_exists = 'append'):
    ## ADD INITIAL DATA TO DATABASE
    engine = create_engine_short(db)
    data.to_sql(sql_table_name, engine, if_exists=if_exists, index = False)
    print('Data successfully saved')
    # RE-QUERY WHOLE DATA
    df = query_database(db, sql_table_name, f"SELECT * FROM {sql_table_name}")

    # DROP DUPLICATES OF WHOLE DATA
    df = df.drop_duplicates()

    # REPLACE INITIAL TABLE WITH NON DUPLICATED TABLE
    df.to_sql(sql_table_name, engine, if_exists='replace', index = False)

def query_database(db, sq_table_name, query):
    engine = create_engine_short(db)
    return pd.read_sql(query, engine)



def create_SQL_connection():

    try:
        connection = mysql.connector.connect(
            host=sql_host,      # The IP address or domain name of your MySQL server (e.g., '192.168.1.100' or 'yourdomain.com')
            port=sql_port,                  # MySQL port number (typically 3306)
            database='securities_master',     # The name of the database you want to connect to
            user= sql_user,         # Your MySQL username
            password=sql_pw     # Your MySQL password
        )
        print("Successfully connected to the database")
    except mysql.connector.Error as err:
        print(err.errno)
        print(err.msg)

    return connection

def close_SQL_connection(connection, cursor = None):
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
    close_SQL_connection(connection, cursor )


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
                raise ValueError(f"Length must be specified for String type column: {col_name}")
            column_type = String(col_length)
        elif col_type == 'Date':
            column_type = Date
        elif col_type == 'Float':
            column_type = Float
        elif col_type == 'Boolean':
            column_type = Boolean
        elif col_type == 'Enum':
            if col_values is None or not isinstance(col_values, list):
                raise ValueError(f"Values must be specified for Enum type column: {col_name}")
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
        print(f"Table '{table_name}' has been created with columns: {[col.name for col in column_definitions]}")
    except SQLAlchemyError as e:
        print(f"An error occurred: {e}")