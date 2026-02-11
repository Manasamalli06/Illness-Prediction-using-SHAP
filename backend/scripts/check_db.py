import os
import sys
import pandas as pd
from sqlalchemy import create_engine, text

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from data.db_config import get_db_connection_string

def check_db():
    try:
        conn_str = get_db_connection_string()
        print(f"Connecting to: {conn_str.split('@')[1]}") # Print host/db only for safety
        
        engine = create_engine(conn_str)
        with engine.connect() as conn:
            print("Connection successful.")
            result = conn.execute(text("SHOW TABLES"))
            tables = [row[0] for row in result]
            print(f"Tables found: {tables}")
            
            if 'patient_records' in tables:
                df = pd.read_sql("SELECT * FROM patient_records LIMIT 5", engine)
                print("Sample data:")
                print(df)
            else:
                print("ERROR: 'patient_records' table not found.")
                
    except Exception as e:
        error_msg = f"Database Error: {e}\n"
        print(error_msg)
        with open("db_log.txt", "w") as f:
            f.write(error_msg)
            import traceback
            traceback.print_exc(file=f)

if __name__ == "__main__":
    check_db()
