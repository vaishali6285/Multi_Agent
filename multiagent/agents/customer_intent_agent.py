import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class CustomerIntentAgent:
    def __init__(self, db_path):
        """
        Initialize the agent with the path to the database.
        """
        self.db_path = db_path
        self.model = None
        self.vectorizer = None

    def fetch_customer_data(self):
        """
        Fetch relevant customer data from the CustomerProfiles table.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT Browsing_History AS browsing_history,
                       Purchase_History AS purchase_history,
                       Customer_Segment AS customer_segment
                FROM CustomerProfiles
            """
            customer_data = pd.read_sql_query(query, conn)
            conn.close()
            return customer_data
        except Exception as e:
            print(f"Error fetching customer data: {e}")
            return pd.DataFrame()

    def train_model(self, customer_data):
        """
        Train a machine learning model for intent prediction.
        """
        print("Training machine learning model...")
        vectorizer = CountVectorizer()

        # Combine browsing and purchase history into one feature column
        combined_features = customer_data['browsing_history'].fillna('') + " " + customer_data['purchase_history'].fillna('')
        features = vectorizer.fit_transform(combined_features)

        # Use customer segment as labels (proxy for intent mapping)
        labels = customer_data['customer_segment']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        # Train a Random Forest classifier
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)

        self.model = clf
        self.vectorizer = vectorizer
        print("Model training completed!")

    def predict_intent(self, browsing, purchase):
        """
        Predict intent using the trained machine learning model.
        """
        if self.model and self.vectorizer:
            combined_feature = browsing + " " + purchase
            features = self.vectorizer.transform([combined_feature])
            return self.model.predict(features)[0]  # Return the predicted label
        return "Unknown Intent"

    def analyze_intent(self, customer_data):
        """
        Analyze customer data to predict intent using rule-based logic and ML.
        """
        intents = []
        for _, row in customer_data.iterrows():
            # Handle potential null values
            browsing = str(row.get('browsing_history', '')).lower()
            purchase = str(row.get('purchase_history', '')).lower()

            # Use machine learning for prediction
            ml_intent = self.predict_intent(browsing, purchase)

            # Rule-Based Intent Logic (extend as needed)
            if 'cart' in browsing or 'checkout' in browsing:
                intents.append('Ready to Purchase')
            elif 'compare' in browsing or 'reviews' in browsing:
                intents.append('Researching Products')
            elif 'gift' in purchase or 'seasonal' in purchase:
                intents.append('Buying Seasonal Gifts')
            elif 'sale' in browsing or 'offer' in browsing:
                intents.append('Looking for Discounts')
            else:
                intents.append(ml_intent)  # Default to ML-based prediction

        customer_data['intent'] = intents
        return customer_data

    def execute(self):
        """
        Main method to execute customer intent analysis.
        """
        print("Starting Customer Intent Analysis...")
        customer_data = self.fetch_customer_data()

        if customer_data.empty:
            print("No customer data available for analysis.")
            return

        # Train the machine learning model
        self.train_model(customer_data)

        # Analyze intents
        intent_data = self.analyze_intent(customer_data)
        print("Customer intents identified successfully!")
        print(intent_data.head())  # Display first few rows for verification
        return intent_data


# Example usage
if __name__ == "__main__":
    # Define the database path
    db_path = r"E:\another\multiagent\db\CustomerpProfiles.db"

    # Create an instance of the Customer Intent Agent and execute its task
    agent = CustomerIntentAgent(db_path)
    intent_data = agent.execute()