import streamlit as st
import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from agents.customer_intent_agent import CustomerIntentAgent # Import the Customer Intent Agent

# Set up database connection
db_path = r'E:\another\multiagent\db\CustomerpProfiles.db'

# Fetch data from the database
def fetch_data(query):
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if there's an error

# Streamlit app
st.title("Customer Insights Dashboard")

# Sidebar for navigation
option = st.sidebar.selectbox("Choose a visualization:", [
    "Customer Segments",
    "Average Order Value by Location",
    "Seasonal Preferences",
    "Customer Count by Location",
    "Customer Intent Distribution"
])

# Visualization options
if option == "Customer Segments":
    query = "SELECT customer_segment, COUNT(*) AS count FROM CustomerProfiles GROUP BY customer_segment"
    data = fetch_data(query)
    if not data.empty:
        st.write("### Customer Distribution by Segment")
        fig, ax = plt.subplots()
        ax.pie(data['count'], labels=data.get('customer_segment', ['Unknown'] * len(data)),
               autopct='%1.1f%%', startangle=90)
        plt.title("Customer Segments")
        st.pyplot(fig)
    else:
        st.warning("No data available for Customer Segments.")

elif option == "Average Order Value by Location":
    query = "SELECT location, AVG(avg_order_value) AS avg_order_value FROM CustomerProfiles GROUP BY location"
    data = fetch_data(query)
    if not data.empty:
        st.write("### Average Order Value by Location")
        fig, ax = plt.subplots()
        sns.barplot(x='location', y='avg_order_value', data=data, ax=ax, palette="viridis")
        ax.set_xlabel("Location")
        ax.set_ylabel("Average Order Value")
        plt.title("Average Order Value by Location")
        st.pyplot(fig)
    else:
        st.warning("No data available for Average Order Value by Location.")

elif option == "Seasonal Preferences":
    query = "SELECT season, COUNT(*) AS count FROM CustomerProfiles GROUP BY season"
    data = fetch_data(query)
    if not data.empty:
        st.write("### Seasonal Preferences")
        fig, ax = plt.subplots()
        sns.barplot(x='season', y='count', data=data, ax=ax, palette="coolwarm")
        ax.set_xlabel("Season")
        ax.set_ylabel("Number of Customers")
        plt.title("Seasonal Preferences")
        st.pyplot(fig)
    else:
        st.warning("No data available for Seasonal Preferences.")

elif option == "Customer Count by Location":
    query = "SELECT location, COUNT(*) AS count FROM CustomerProfiles GROUP BY location"
    data = fetch_data(query)
    if not data.empty:
        st.write("### Customer Count by Location")
        fig, ax = plt.subplots()
        sns.barplot(x='location', y='count', data=data, ax=ax, palette="magma")
        ax.set_xlabel("Location")
        ax.set_ylabel("Customer Count")
        plt.title("Customer Count by Location")
        st.pyplot(fig)
    else:
        st.warning("No data available for Customer Count by Location.")

elif option == "Customer Intent Distribution":
    # Create an instance of the Customer Intent Agent
    intent_agent = CustomerIntentAgent(db_path)

    # Execute the agent to fetch and analyze customer intents
    intent_data = intent_agent.execute()

    if not intent_data.empty:
        st.write("### Customer Intent Distribution")
        # Visualize intent distribution
        intent_counts = intent_data['intent'].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=intent_counts.index, y=intent_counts.values, ax=ax, palette="magma")
        ax.set_xlabel("Intent Categories")
        ax.set_ylabel("Number of Customers")
        ax.set_title("Distribution of Customer Intents")
        st.pyplot(fig)

        # Allow filtering by intent
        st.write("### Filter Customers by Intent")
        selected_intent = st.selectbox("Select an Intent to Filter Customers:",
                                       intent_data['intent'].unique())
        filtered_data = intent_data[intent_data['intent'] == selected_intent]
        st.write(f"#### Customers with Intent: {selected_intent}")
        st.dataframe(filtered_data)
    else:
        st.warning("No customer intent data available to visualize.")