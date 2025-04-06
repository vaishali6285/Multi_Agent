import sqlite3
import os

db_path = 'E:/another/multiagent/db/multi_agent_system.db'  # Correct path yahan daaliye
print("Database Path:", os.path.abspath(db_path))

try:
    conn = sqlite3.connect(db_path)
    print("Database se connection successful!")

    # SQLite Version Check
    cursor = conn.cursor()
    cursor.execute("SELECT sqlite_version();")
    sqlite_version = cursor.fetchone()
    print("SQLite Version:", sqlite_version)

    conn.close()
    print("Connection close ho gaya.")
except sqlite3.Error as e:
    print(f"Database connection error: {e}")