import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
import random


class PersonalizationExperienceAgent:
    def __init__(self, database_path, item_mapping_path):
        """
        Initialize the agent with access to user database and item mapping.
        """
        self.database_path = database_path
        self.item_mapping_path = item_mapping_path
        self.user_data = None
        self.segmented_users = None
        self.nmf_model = None
        self.recommendations = {}

    def fetch_user_data(self):
        """
        Fetch user data from the database.
        """
        print("Fetching user data...")
        try:
            self.user_data = pd.read_csv(self.database_path)
            print("User data successfully fetched!")
        except Exception as e:
            print(f"Error fetching user data: {e}")

    def segment_users(self):
        """
        Dynamically segment users based on behavior or demographic trends.
        """
        print("Segmenting users...")
        if self.user_data is not None:
            bins = [-1, 10, 30, float('inf')]  # Define bins in increasing order
            labels = ['Occasional Shoppers', 'Explorers', 'Frequent Buyers']

            # Segment users based on purchase frequency
            self.user_data['user_segment'] = pd.cut(
                self.user_data['purchase_frequency'], bins=bins, labels=labels, right=True
            )

            # Save segmentation results
            self.segmented_users = self.user_data[['user_id', 'user_segment']]
            print("User segmentation completed!")
        else:
            print("User data is not available. Please fetch the data first.")

    def build_recommendation_model(self):
        """
        Train a recommendation model using Non-Negative Matrix Factorization (NMF).
        """
        print("Building recommendation model...")
        if self.user_data is not None:
            # Prepare interaction matrix (e.g., user-item matrix for recommendations)
            interaction_matrix = self.user_data.pivot(index='user_id', columns='item_id',
                                                      values='interaction_score').fillna(0)

            # Ensure non-negative values in the interaction matrix
            interaction_matrix[interaction_matrix < 0] = 0

            # Scale data using MinMaxScaler to ensure values are in [0, 1]
            scaler = MinMaxScaler()
            interaction_matrix_scaled = scaler.fit_transform(interaction_matrix)

            # Train NMF model
            self.nmf_model = NMF(n_components=5, random_state=42)
            self.nmf_model.fit(interaction_matrix_scaled)
            print("Recommendation model built successfully!")
        else:
            print("User data is not available. Please fetch the data first.")

    def generate_recommendations(self, user_id):
        """
        Generate personalized recommendations for a given user using the NMF model.
        """
        print(f"Generating recommendations for User ID: {user_id}...")
        if self.user_data is not None and self.nmf_model is not None:
            # Get the interaction matrix
            interaction_matrix = self.user_data.pivot(index='user_id', columns='item_id',
                                                      values='interaction_score').fillna(0)

            # Predict scores for all items
            user_index = list(interaction_matrix.index).index(user_id)
            user_scores = self.nmf_model.inverse_transform(self.nmf_model.transform(interaction_matrix))

            # Rank items by predicted score
            recommended_item_ids = interaction_matrix.columns[user_scores[user_index].argsort()[::-1][:5]].tolist()

            # Map item IDs to item names using the item mapping dataset
            try:
                item_mapping = pd.read_csv(self.item_mapping_path)  # Load item mapping
                item_name_mapping = dict(zip(item_mapping["item_id"], item_mapping["item_name"]))
                recommended_items = [item_name_mapping.get(item_id, f"Item {item_id}") for item_id in
                                     recommended_item_ids]
            except Exception as e:
                print(f"Error loading item mapping: {e}")
                recommended_items = recommended_item_ids  # Fallback to IDs if mapping fails

            # Store the recommendations
            self.recommendations[user_id] = recommended_items
            print(f"Recommendations for User ID {user_id}: {recommended_items}")
        else:
            print(
                "User data or recommendation model is not available. Please fetch the data and build the model first.")

    def create_personalized_message(self, user_id):
        """
        Create a personalized message or offer for the user.
        """
        print(f"Creating personalized message for User ID: {user_id}...")
        if user_id in self.recommendations:
            items = self.recommendations[user_id]
            message = f"Hello, User {user_id}! Based on your preferences, we recommend you check out these items: {', '.join(map(str, items))}."
            print("Message created successfully!")
            return message
        else:
            print("Recommendations not available for this user.")
            return None


# Example Usage
if __name__ == "__main__":
    # Example CSV Paths (replace with actual database or data files)
    database_path = "cleared_customer_data.csv"
    item_mapping_path = "item_mapping.csv"

    # Sample dataset creation for demonstration purposes
    customer_data = pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5],
        'item_id': [101, 102, 103, 104, 105],
        'interaction_score': [5, 3, 4, 2, 1],
        'purchase_frequency': [12, 7, 10, 5, 15],
        'browsing_time': [35, 15, 25, 10, 50]
    })
    item_mapping_data = pd.DataFrame({
        'item_id': [101, 102, 103, 104, 105],
        'item_name': ['Smartphone', 'Laptop', 'Headphones', 'Tablet', 'Smartwatch']
    })
    customer_data.to_csv(database_path, index=False)  # Save sample customer data as CSV
    item_mapping_data.to_csv(item_mapping_path, index=False)  # Save sample item mapping as CSV

    agent = PersonalizationExperienceAgent(database_path, item_mapping_path)

    # Step 1: Fetch user data
    agent.fetch_user_data()

    # Step 2: Segment users
    agent.segment_users()

    # Step 3: Build recommendation model
    agent.build_recommendation_model()

    # Step 4: Generate recommendations for a user
    agent.generate_recommendations(user_id=1)

    # Step 5: Create a personalized message for the user
    message = agent.create_personalized_message(user_id=1)
    print("Personalized Message:", message)