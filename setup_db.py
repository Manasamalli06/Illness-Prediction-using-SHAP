import pymysql
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from data.db_config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT

def create_database():
    try:
        print(f"Connecting to {DB_HOST}...")
        conn = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            port=int(DB_PORT)
        )
        cursor = conn.cursor()
        print(f"Creating database '{DB_NAME}' if not exists...")
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
        print("Database created (or already exists).")
        conn.close()
    except Exception as e:
        print(f"Error creating database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    create_database()
