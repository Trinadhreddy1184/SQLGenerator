# utils/file_handler.py

import os
import pandas as pd

import pyarrow.parquet as pq
import openpyxl
import fastavro


def parse_file(file_path: str):
    """
    Parses a file into a pandas DataFrame based on its file extension.
    Supported formats: CSV, TSV, TXT, Parquet, Avro.
    Returns DataFrame or None if an error occurs or the format is unsupported.
    """
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext in [".csv", ".tsv", ".txt"]:
            sep = "\t" if ext == ".tsv" else ","
            df = pd.read_csv(file_path, sep=sep)
        elif ext == ".xlsx":
            return pd.read_excel(file_path, engine='openpyxl')

        elif ext == ".parquet":
            if not pq:
                raise ImportError("pyarrow is required to read .parquet files")
            df = pd.read_parquet(file_path)

        elif ext == ".avro":
            if not fastavro:
                raise ImportError("fastavro is required to read .avro files")
            with open(file_path, "rb") as fo:
                reader = fastavro.reader(fo)
                records = list(reader)
                df = pd.DataFrame(records)

        else:
            raise ValueError(f"Unsupported file format: {ext}")

        return df

    except Exception as e:
        print(f"[FileHandler ERROR]: {e}")
        return None