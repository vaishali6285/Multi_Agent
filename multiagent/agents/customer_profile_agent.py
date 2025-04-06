import sqlite3
import os


# Step 1: Create the SQLite database and CustomerProfiles table
def create_database(db_path):
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Create CustomerProfiles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS CustomerProfiles (
                customer_id INTEGER PRIMARY KEY,
                browsing_history TEXT,
                purchase_history TEXT,
                customer_segment TEXT,
                location TEXT,
                age INTEGER,
                avg_order_value REAL
            )
        """)
        conn.commit()
        print("CustomerProfiles table created successfully!")
        conn.close()
    except sqlite3.Error as e:
        print(f"Database creation error: {e}")


# Step 2: Insert sample data into the CustomerProfiles table
def insert_sample_data(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if data already exists to avoid duplicate insertions
        cursor.execute("SELECT COUNT(*) FROM CustomerProfiles")
        count = cursor.fetchone()[0]

        if count == 0:
            # Insert sample customer data
            cursor.execute("""
                INSERT INTO CustomerProfiles (customer_id, browsing_history, purchase_history, customer_segment, location, age, avg_order_value)
                VALUES
                    (1, 'Books,Fashion', 'Fashion Accessories', 'Premium', 'New Delhi', 25, 500.0),
                    (2, 'Electronics', 'Smartphones', 'Regular', 'Mumbai', 30, 1200.0),
                    (3, 'Home Decor', 'Furniture', 'Premium', 'Chennai', 35, 2000.0),
                    (4, 'Fashion', 'Clothing', 'Regular', 'Bangalore', 28, 700.0),
                    (5, 'Electronics,Books', 'Laptops', 'Premium', 'Hyderabad', 40, 1500.0)
            """)
            conn.commit()
            print("Sample data inserted successfully!")
        else:
            print("Data already exists, skipping insertion.")

        conn.close()
    except sqlite3.Error as e:
        print(f"Data insertion error: {e}")


# Step 3: Define the CustomerProfileAgent to interact with the database
class CustomerProfileAgent:
    def __init__(self, db_path):
        self.db_path = db_path

        # Ensure database exists
        if not os.path.exists(db_path):
            create_database(db_path)
            insert_sample_data(db_path)

    def connect_to_db(self):
        try:
            conn = sqlite3.connect(self.db_path)
            return conn
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            return None

    def fetch_all_customers(self):
        conn = self.connect_to_db()
        if conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT * FROM CustomerProfiles")
                # Convert to list of dictionaries for better usability
                columns = [description[0] for description in cursor.description]
                customers = [dict(zip(columns, row)) for row in cursor.fetchall()]
                conn.close()
                return customers
            except sqlite3.Error as e:
                print(f"Error fetching customer profiles: {e}")
                conn.close()
                return []
        else:
            print("Failed to connect to the database.")
            return []

    def fetch_customer_by_id(self, customer_id):
        conn = self.connect_to_db()
        if conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT * FROM CustomerProfiles WHERE customer_id = ?", (customer_id,))
                row = cursor.fetchone()
                if row:
                    # Convert to dictionary for better usability
                    columns = [description[0] for description in cursor.description]
                    customer = dict(zip(columns, row))
                    conn.close()
                    return customer
                else:
                    conn.close()
                    return None
            except sqlite3.Error as e:
                print(f"Error fetching customer profile: {e}")
                conn.close()
                return None
        else:
            print("Failed to connect to the database.")
            return None


# Example Usage
if __name__ == "__main__":
    # Use a relative path for better portability
    db_path = os.path.join(os.path.dirname(__file__), "db", "multi_agent_system.db")

    # Create agent instance
    agent = CustomerProfileAgent(db_path)

    print("\nFetching all customer profiles...")
    all_customers = agent.fetch_all_customers()
    for customer in all_customers:
        print(f"Customer: {customer}")

    print("\nFetching customer with ID 1...")
    customer = agent.fetch_customer_by_id(1)
    if customer:
        print("Customer 1:", customer)
    else:
        print("Customer not found.")