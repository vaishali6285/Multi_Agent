import pandas as pd
import sqlite3

csv_path = r'data/customer_data_collection.csv'
db_path = r'E:\another\multiagent\db\CustomerpProfiles.db'

def import_csv_to_db(csv_path, db_path):
    try:
        df = pd.read_csv(csv_path)
        conn = sqlite3.connect(db_path)
        df.to_sql('CustomerProfiles', conn, if_exists='replace', index=False)
        print("CSV data imported successfully!")
        conn.close()
    except Exception as e:
        print(f"Error importing CSV to database: {e}")

import_csv_to_db(csv_path, db_path)