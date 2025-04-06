import sqlite3

# Path to your SQLite database file
db_path = 'E:/another/multiagent/db/multi_agent_system.db'

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Insert sample data into the CustomerProfiles table
    cursor.execute('''
        INSERT INTO CustomerProfiles (customer_id, age, gender, location, browsing_history, purchase_history, customer_segment, avg_order_value, holiday, season)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        1000,  # INTEGER for customer_id (no quotes because it's INTEGER)
        25,  # INTEGER for age
        'Female',  # TEXT for gender
        'New Delhi',  # TEXT for location
        '["Electronics", "Books"]',  # TEXT for browsing_history
        '["Laptop", "Headphones"]',  # TEXT for purchase_history
        'Premium',  # TEXT for customer_segment
        1500.75,  # FLOAT for avg_order_value
        'Christmas',  # TEXT for holiday
        'Winter'  # TEXT for season
    ))
    conn.commit()
    print("Sample data inserted successfully into CustomerProfiles!")

    conn.close()
except sqlite3.Error as e:
    print(f"Error inserting data into CustomerProfiles: {e}")