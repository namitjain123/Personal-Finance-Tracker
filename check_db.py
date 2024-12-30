import sqlite3
from db_config import DB_PATH

def check_tables():
    try:
        connection = sqlite3.connect(DB_PATH)
        cursor = connection.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print("Available tables:")
        for table in tables:
            print(f"- {table[0]}")
            
        # Show table structures
        for table in tables:
            print(f"\nStructure of {table[0]}:")
            cursor.execute(f"PRAGMA table_info({table[0]})")
            columns = cursor.fetchall()
            for column in columns:
                print(f"  {column[1]} ({column[2]})")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        connection.close()

if __name__ == "__main__":
    check_tables() 