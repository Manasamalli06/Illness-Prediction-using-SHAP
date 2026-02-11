import os

# Database Configuration
# You can change these values or set environment variables
DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_USER = os.environ.get('DB_USER', 'root')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'manasa') # UPDATE THIS
DB_NAME = os.environ.get('DB_NAME', 'healthcare_prediction')
DB_PORT = os.environ.get('DB_PORT', '3306')

def get_db_connection_string():
    return f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
