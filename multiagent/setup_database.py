import sqlite3
import pandas as pd

# Load cleaned datasets
customer_data = pd.read_csv('data/cleaned_customer_data.csv')
product_data = pd.read_csv('data/cleaned_product_data.csv')

# Connect to SQLite (creates the database file if it doesn't exist)
conn = sqlite3.connect('db/multi_agent_system.db')
cursor = conn.cursor()

# Create CustomerProfiles table with customer_id as TEXT
cursor.execute('''
    CREATE TABLE IF NOT EXISTS CustomerProfiles (
        customer_id TEXT PRIMARY KEY,
        age INTEGER,
        gender TEXT,
        location TEXT,
        browsing_history TEXT,
        purchase_history TEXT,
        customer_segment TEXT,
        avg_order_value FLOAT,
        holiday TEXT,
        season TEXT
    )
''')

# Create Products table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS Products (
        product_id INTEGER PRIMARY KEY,
        category TEXT,
        subcategory TEXT,
        price FLOAT,
        brand TEXT,
        avg_rating FLOAT,
        sentiment_score FLOAT,
        holiday TEXT,
        season TEXT,
        similar_products TEXT,
        recommendation_probability FLOAT
    )
''')

# Insert data into CustomerProfiles table
for _, row in customer_data.iterrows():
    try:
        # Handle potential missing values and cast to correct types
        customer_id = str(row['Customer_ID']) if pd.notnull(row['Customer_ID']) else "Unknown"
        age = int(row['Age']) if pd.notnull(row['Age']) and str(row['Age']).isdigit() else 0
        gender = str(row['Gender']) if pd.notnull(row['Gender']) else "Unknown"
        location = str(row['Location']) if pd.notnull(row['Location']) else "Unknown"
        browsing_history = str(row['Browsing_History']) if pd.notnull(row['Browsing_History']) else "Unknown"
        purchase_history = str(row['Purchase_History']) if pd.notnull(row['Purchase_History']) else "Unknown"
        customer_segment = str(row['Customer_Segment']) if pd.notnull(row['Customer_Segment']) else "Unknown"
        avg_order_value = float(row['Avg_Order_Value']) if pd.notnull(row['Avg_Order_Value']) else 0.0
        holiday = str(row['Holiday']) if pd.notnull(row['Holiday']) else "Unknown"
        season = str(row['Season']) if pd.notnull(row['Season']) else "Unknown"

        cursor.execute('''
            INSERT OR IGNORE INTO CustomerProfiles 
            (customer_id, age, gender, location, browsing_history, purchase_history, customer_segment, avg_order_value, holiday, season)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            customer_id, age, gender, location, browsing_history, purchase_history,
            customer_segment, avg_order_value, holiday, season
        ))
    except Exception as e:
        print(f"Error inserting row into CustomerProfiles: {e}")
        continue

# Insert data into Products table
for _, row in product_data.iterrows():
    try:
        # Handle potential missing values and cast to correct types
        product_id = int(row['product_id']) if pd.notnull(row['product_id']) and str(row['product_id']).isdigit() else 0
        category = str(row['category']) if pd.notnull(row['category']) else "Unknown"
        subcategory = str(row['subcategory']) if pd.notnull(row['subcategory']) else "Unknown"
        price = float(row['price']) if pd.notnull(row['price']) else 0.0
        brand = str(row['brand']) if pd.notnull(row['brand']) else "Unknown"
        avg_rating = float(row['avg_rating']) if pd.notnull(row['avg_rating']) else 0.0
        sentiment_score = float(row['sentiment_score']) if pd.notnull(row['sentiment_score']) else 0.0
        holiday = str(row['holiday']) if pd.notnull(row['holiday']) else "Unknown"
        season = str(row['season']) if pd.notnull(row['season']) else "Unknown"
        similar_products = str(row['similar_products']) if pd.notnull(row['similar_products']) else "Unknown"
        recommendation_probability = float(row['recommendation_probability']) if pd.notnull(row['recommendation_probability']) else 0.0

        cursor.execute('''
            INSERT OR IGNORE INTO Products 
            (product_id, category, subcategory, price, brand, avg_rating, sentiment_score, holiday, season, similar_products, recommendation_probability)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            product_id, category, subcategory, price, brand, avg_rating, sentiment_score,
            holiday, season, similar_products, recommendation_probability
        ))
    except Exception as e:
        print(f"Error inserting row into Products: {e}")
        continue

# Commit the transaction and close the connection
conn.commit()
conn.close()

print("Database setup completed successfully!")