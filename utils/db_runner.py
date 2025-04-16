# utils/db_runner.py

import duckdb
import pandas as pd

# In-memory DuckDB connection
conn = duckdb.connect(database=':memory:')
conn.execute("PRAGMA enable_object_cache")

# Track loaded tables and schemas
registered_tables = {}

def load_table(df: pd.DataFrame, table_name: str):
    """
    Registers a DataFrame as a DuckDB table with the given name.
    """
    global registered_tables
    try:
        conn.register(table_name, df)
        registered_tables[table_name] = df.columns.tolist()
        print(f"[DB] Loaded table '{table_name}' with columns: {df.columns.tolist()}")
        return True
    except Exception as e:
        print(f"[DB ERROR]: {e}")
        return False

def get_schema_overview():
    """
    Returns a formatted string summarizing all registered table schemas.
    """
    schema_texts = []
    for table, cols in registered_tables.items():
        schema_texts.append(f"Table `{table}`: Columns → {', '.join(cols)}")
    return "\n".join(schema_texts)

def describe_all_tables():
    tables = conn.execute("SHOW TABLES").fetchall()
    combined = []

    for (table,) in tables:
        desc = conn.execute(f"DESCRIBE {table}").fetchdf()
        desc["table"] = table
        combined.append(desc)

    return pd.concat(combined, ignore_index=True)

def run_query(sql: str, limit: int = 20):
    """
    Executes a SQL query and returns up to `limit` rows formatted in markdown.
    """
    try:
        limited_sql = f"SELECT * FROM ({sql}) LIMIT {limit}"
        result_df = conn.execute(limited_sql).fetchdf()
        return result_df
    except Exception as e:
        return f"⚠️ SQL Execution Error: {str(e)}"
