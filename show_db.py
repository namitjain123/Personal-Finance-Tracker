import sqlite3
from pathlib import Path
import os

def display_table_contents():
    """Display contents of all tables in the expense tracker database"""
    
    # Get the database path - assuming it's in the same directory
    db_path = Path('data/expense_tracker.db')
    
    if not db_path.exists():
        print(f"Database not found at: {db_path.absolute()}")
        return
        
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            print("No tables found in database")
            return
            
        # Display contents of each table
        for table in tables:
            table_name = table[0]
            print("\n" + "="*50)
            print(f"Contents of table: {table_name}")
            print("="*50)
            
            # Get column names
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]
            print("Columns:", " | ".join(column_names))
            print("-"*50)
            
            # Get all rows
            cursor.execute(f"SELECT * FROM {table_name};")
            rows = cursor.fetchall()
            
            if not rows:
                print("No data in table")
            else:
                # Print each row
                for row in rows:
                    formatted_row = [str(item) for item in row]
                    print(" | ".join(formatted_row))
                print(f"\nTotal rows: {len(rows)}")
                
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    print("Personal Expense Tracker - Database Contents")
    print("="*50)
    display_table_contents()