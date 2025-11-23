import mysql.connector

# Centralized Database Configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "",
    "password": "",
    "database": "smart_attendance"
}

def init_db():
    """
    SAFE SETUP: Creates Database and Table ONLY if they don't exist.
    This function will NOT delete or overwrite your existing data.
    """
    try:
        # 1. Connect to MySQL Server (Root level, no DB selected yet)
        conn = mysql.connector.connect(
            host=DB_CONFIG["host"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"]
        )
        cursor = conn.cursor()

        # 2. Create Database (Safe: Does nothing if DB exists)
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
        
        # 3. Select the Database
        cursor.execute(f"USE {DB_CONFIG['database']}")

        # 4. Create Table (Safe: Does nothing if Table exists)
        # Note: We do NOT use TRUNCATE here, so data is preserved.
        create_table_query = """
        CREATE TABLE IF NOT EXISTS attendance (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            date DATE NOT NULL,
            time TIME NOT NULL,
            UNIQUE KEY uniq_name_date (name, date)
        );
        """
        cursor.execute(create_table_query)
        
        conn.commit()
        cursor.close()
        conn.close()
        # print("✅ Database check complete (Data preserved).")
        
    except mysql.connector.Error as err:
        print(f"❌ Database Initialization Error: {err}")

def get_connection():
    """
    Helper function for app.py to get a connection easily.
    """
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as err:
        # st.error(f"Database Connection Failed: {err}")
        return None

if __name__ == "__main__":
    # This allows you to run 'python db.py' manually if you ever want to
    init_db()
    print("Manual setup complete.")