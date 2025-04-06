import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

# Database path
db_path = 'E:/another/multiagent/db/CustomerpProfiles.db'  # Update this path as needed

# Connect to the database
def connect_to_db():
    try:
        conn = sqlite3.connect(db_path)
        print("Database connection successful!")
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        return None

# Fetch data for customer segments
def fetch_segment_data():
    conn = connect_to_db()
    if conn:
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT customer_segment, COUNT(*) FROM CustomerProfiles GROUP BY customer_segment")
            segment_data = cursor.fetchall()
            conn.close()
            return segment_data
        except sqlite3.Error as e:
            print(f"Error fetching segment data: {e}")
            conn.close()
            return []
    else:
        print("Failed to connect to the database.")
        return []

# Fetch data for average order value by location
def fetch_avg_order_value_by_location():
    conn = connect_to_db()
    if conn:
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT location, AVG(avg_order_value) FROM CustomerProfiles GROUP BY location")
            location_data = cursor.fetchall()
            conn.close()
            return location_data
        except sqlite3.Error as e:
            print(f"Error fetching average order value by location: {e}")
            conn.close()
            return []
    else:
        print("Failed to connect to the database.")
        return []

# Fetch data for seasonal preferences
def fetch_season_data():
    conn = connect_to_db()
    if conn:
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT season, COUNT(*) FROM CustomerProfiles GROUP BY season")
            season_data = cursor.fetchall()
            conn.close()
            return season_data
        except sqlite3.Error as e:
            print(f"Error fetching season data: {e}")
            conn.close()
            return []
    else:
        print("Failed to connect to the database.")
        return []

# Generate pie chart for customer segments
def visualize_segments():
    data = fetch_segment_data()
    if data:
        segments = [row[0] for row in data]  # Extract segment names
        counts = [row[1] for row in data]   # Extract counts
        plt.figure(figsize=(8, 6))
        plt.pie(counts, labels=segments, autopct='%1.1f%%', startangle=90)
        plt.title("Customer Distribution by Segment")
        plt.axis('equal')  # Equal aspect ratio ensures the pie is circular
        plt.show()
    else:
        print("No data available for visualization.")

# Generate bar chart for average order value by location
def visualize_avg_order_value_by_location():
    data = fetch_avg_order_value_by_location()
    if data:
        locations = [row[0] for row in data]  # Extract location names
        avg_values = [row[1] for row in data]  # Extract average order values
        plt.figure(figsize=(10, 6))
        sns.barplot(x=locations, y=avg_values, palette="viridis")
        plt.title("Average Order Value by Location")
        plt.xlabel("Location")
        plt.ylabel("Average Order Value")
        plt.xticks(rotation=45)
        plt.show()
    else:
        print("No data available for visualization.")

# Generate bar chart for seasonal preferences
def visualize_seasonal_preferences():
    data = fetch_season_data()
    if data:
        seasons = [row[0] for row in data]  # Extract season names
        counts = [row[1] for row in data]  # Extract customer counts
        plt.figure(figsize=(10, 6))
        sns.barplot(x=seasons, y=counts, palette="coolwarm")
        plt.title("Seasonal Preferences")
        plt.xlabel("Season")
        plt.ylabel("Number of Customers")
        plt.xticks(rotation=45)
        plt.show()
    else:
        print("No data available for visualization.")

# Example usage
if __name__ == "__main__":
    print("\nVisualizing Customer Segments...")
    visualize_segments()

    print("\nVisualizing Average Order Value by Location...")
    visualize_avg_order_value_by_location()

    print("\nVisualizing Seasonal Preferences...")
    visualize_seasonal_preferences()