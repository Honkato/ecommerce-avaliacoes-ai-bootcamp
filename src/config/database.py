import sqlite3
import csv
import os

DB_FILE = os.path.join(os.path.dirname(__file__), "..", "reviews.db")


def get_connection():

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    return conn, cursor


def csv_to_sqlite(csv_filepath: str):

    table_name = "reviews"

    if not os.path.isfile(csv_filepath):
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {csv_filepath}")

    with open(csv_filepath, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError("O CSV está vazio.")

        data = list(reader)

    conn, cursor = get_connection()
    try:
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.commit()

        cols_quoted = [f'"{col}" TEXT' for col in header]
        create_stmt = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(cols_quoted)});"
        cursor.execute(create_stmt)
        conn.commit()

        placeholders = ", ".join(["?"] * len(header))
        insert_stmt = f"INSERT INTO {table_name} ({', '.join([f'\"{col}\"' for col in header])}) VALUES ({placeholders});"

        cursor.executemany(insert_stmt, data)
        conn.commit()

    finally:
        cursor.close()
        conn.close()


def select_distinct_all():
    conn, cursor = get_connection()
    try:
        query = "SELECT DISTINCT * FROM reviews;"
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    finally:
        cursor.close()
        conn.close()


def select_distinct_by_column(columns: str):

    conn, cursor = get_connection()
    try:

        query = f"SELECT DISTINCT {columns} FROM reviews;"
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    finally:
        cursor.close()
        conn.close()
