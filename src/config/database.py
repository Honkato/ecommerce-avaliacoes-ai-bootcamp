import sqlite3
import csv
import sqlite3
import os

db_file = './my_database.db'

def setup_db():
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    return cursor


def select_distinct_by_column(columns):
    cursor = setup_db()
    query = f"SELECT DISTINCT {columns} FROM reviews"
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    return result

def select_distinct_all():
    cursor = setup_db()
    query = f"SELECT DISTINCT * FROM reviews"
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    return result

def csv_to_sqlite(csv_filepath):
    """Imports a CSV file into a SQLite database table.

    Args:
        csv_filepath: Path to the CSV file.
    """
    table_name = "reviews"
    try:
        with open(csv_filepath, 'r') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)
            data = list(csv_reader)

        with sqlite3.connect(db_file) as connection:
            cursor = connection.cursor()

            # Create table if it doesn't exist
            placeholders = ', '.join(['?'] * len(header))
            create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(header)});"
            cursor.execute(create_table_query)

            # Insert data into table
            insert_query = f"INSERT INTO {table_name} VALUES ({placeholders});"
            cursor.executemany(insert_query, data)

            connection.commit()

    except Exception as e:
        print(f"An error occurred: {e}")
