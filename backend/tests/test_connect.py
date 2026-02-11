import pymysql
import sys
import os

# Add backend to path to import data.db_config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.append(BACKEND_DIR)

from data.db_config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT

try:
    print(f"Connecting to {DB_HOST} as {DB_USER}...")
    conn = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        port=int(DB_PORT)
    )
    print("Combined Connection Successful (Server reachable)")
    
    cursor = conn.cursor()
    cursor.execute(f"SHOW DATABASES LIKE '{DB_NAME}'")
    result = cursor.fetchone()
    
    if result:
        print(f"Database '{DB_NAME}' EXISTS.")
    else:
        print(f"Database '{DB_NAME}' DOES NOT EXIST.")
        sys.exit(2) # 2 for Missing DB
        
    conn.select_db(DB_NAME)
    print("Select DB Successful")
    
    cursor.execute("SHOW TABLES")
    tables = [r[0] for r in cursor.fetchall()]
    print(f"Tables: {tables}")
    conn.close()
    
except pymysql.err.OperationalError as e:
    print(f"Operational Error: {e}")
    if e.args[0] == 1045:
        print("ACCESS DENIED (Wrong Password/User)")
        sys.exit(3) # 3 for Auth Fail
    elif e.args[0] == 1049:
        print("UNKNOWN DATABASE")
        sys.exit(2)
    else:
        sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
