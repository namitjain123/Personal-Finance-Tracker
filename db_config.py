import os
import sqlite3
import sys

def get_db_path():
    # Get the directory where the executable/script is located
    if getattr(sys, 'frozen', False):
        application_path = os.path.dirname(sys.executable)
    else:
        application_path = os.path.dirname(os.path.abspath(__file__))
    
    # Create a data directory if it doesn't exist
    data_dir = os.path.join(application_path, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    return os.path.join(data_dir, 'expense_tracker.db')

DB_PATH = get_db_path()

def initialize_database():
    """Create database and tables if they don't exist"""
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            income REAL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            date DATE NOT NULL,
            category TEXT NOT NULL,
            amount REAL NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users(username)
        )
    """)
    
    connection.commit()
    connection.close()

def get_db_connection():
    initialize_database()  # Ensure database exists
    try:
        connection = sqlite3.connect(DB_PATH)
        connection.row_factory = sqlite3.Row
        return connection
    except Exception as e:
        print(f"Error connecting to SQLite: {e}")
        return None 