import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler


class RecommendationEngineAgent:
    def __init__(self, user_data_path, product_data_path):
        """
        Initialize the agent with access to user and product datasets.
        """
        self.user_data_path = user_data_path
        self.product_data_path = product_data_path
        self.user_data = None
        self.product_data = None
        self.user_product_matrix = None
        self.nmf_model = None
        self.recommendations = {}

    def fetch_data(self):
        """
        Fetch user and product data from provided datasets.
        """
        print("Fetching user and product data...")
        try:
            self.user_data = pd.read_csv(self.user_data_path)
            self.product_data = pd.read_csv(self.product_data_path)

            # Display the actual column names to help with debugging
            print("User Data Columns:", self.user_data.columns.tolist())
            print("Product Data Columns:", self.product_data.columns.tolist())

            # Basic validation and sampling of the data
            print("Data successfully fetched!")
            print("User Data Sample:")
            print(self.user_data.head())
            print("Product Data Sample:")
            print(self.product_data.head())

            return True
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False

    def create_synthetic_interaction_data(self):
        """
        Create synthetic user-product interaction data since the original data
        doesn't seem to have the right structure.
        """
        print("Creating synthetic user-product interaction data...")

        if self.user_data is None or self.product_data is None:
            print("User or product data not available. Please fetch the data first.")
            return False

        try:
            # Get user IDs - use Customer_ID if available, otherwise use index
            if 'Customer_ID' in self.user_data.columns:
                user_ids = self.user_data['Customer_ID'].tolist()
            else:
                print("Warning: No Customer_ID column found, using index as user ID")
                user_ids = self.user_data.index.tolist()

            # Get product IDs
            if 'Product_ID' in self.product_data.columns:
                product_ids = self.product_data['Product_ID'].tolist()
            else:
                print("Warning: No Product_ID column found, using index as product ID")
                product_ids = self.product_data.index.tolist()

            # Create a synthetic user-product interaction dataframe
            # Each user will interact with a random subset of products
            interactions = []

            for user_id in user_ids:
                # Randomly select 3-10 products for each user
                num_products = np.random.randint(3, min(10, len(product_ids) + 1))
                selected_products = np.random.choice(product_ids, size=num_products, replace=False)

                for product_id in selected_products:
                    # Generate random interaction score (1-5)
                    score = np.random.randint(1, 6)
                    interactions.append({
                        'User_ID': user_id,
                        'Product_ID': product_id,
                        'Interaction_Score': score
                    })

            # Create DataFrame from interactions
            self.user_product_interactions = pd.DataFrame(interactions)
            print("Synthetic user-product interaction data created successfully!")
            print("Interaction Data Sample:")
            print(self.user_product_interactions.head())

            return True
        except Exception as e:
            print(f"Error creating synthetic interaction data: {e}")
            return False

    def build_recommendation_model(self):
        """
        Train a recommendation model using Non-Negative Matrix Factorization (NMF).
        """
        print("Building recommendation model...")

        # Check if we have user-product interaction data
        if not hasattr(self, 'user_product_interactions') or self.user_product_interactions is None:
            print("No user-product interaction data available.")
            if self.create_synthetic_interaction_data():
                print("Created synthetic data, proceeding with model building.")
            else:
                print("Failed to create interaction data. Cannot build model.")
                return False

        try:
            # Prepare interaction matrix
            interaction_matrix = self.user_product_interactions.pivot(
                index='User_ID', columns='Product_ID', values='Interaction_Score'
            ).fillna(0)

            # Store this for later use
            self.user_product_matrix = interaction_matrix

            print("Interaction Matrix Shape:", interaction_matrix.shape)
            print("Interaction Matrix Sample:")
            print(interaction_matrix.head())

            # Scale data using MinMaxScaler
            scaler = MinMaxScaler()
            interaction_matrix_scaled = scaler.fit_transform(interaction_matrix)

            # Train NMF model
            self.nmf_model = NMF(n_components=min(5, min(interaction_matrix.shape) - 1),
                                 random_state=42, max_iter=500)
            self.nmf_model.fit(interaction_matrix_scaled)
            print("Recommendation model built successfully!")
            return True
        except Exception as e:
            print(f"Error building recommendation model: {e}")
            return False

    def generate_recommendations(self, user_id, price_range=None, category=None):
        """
        Generate personalized recommendations for a given user with optional filters.
        """
        print(f"Generating recommendations for User_ID: {user_id}...")

        if self.nmf_model is None:
            print("Recommendation model not available. Building model now...")
            if not self.build_recommendation_model():
                print("Failed to build recommendation model.")
                return False

        try:
            # Check if user exists in the matrix
            if user_id not in self.user_product_matrix.index:
                print(f"User_ID {user_id} not found in user data.")
                print("Available user IDs:", self.user_product_matrix.index.tolist()[:10], "...")
                self.recommendations[user_id] = []
                return False

            # Get user index in the matrix
            user_index = list(self.user_product_matrix.index).index(user_id)

            # Get user's latent features
            user_vector = self.nmf_model.transform(self.user_product_matrix)[user_index:user_index + 1]

            # Get predicted scores for all products
            predicted_scores = self.nmf_model.inverse_transform(user_vector)[0]

            # Get product IDs ordered by predicted score
            product_indices = predicted_scores.argsort()[::-1]
            recommended_product_ids = [self.user_product_matrix.columns[i] for i in product_indices[:20]]

            print(f"Recommended Product IDs: {recommended_product_ids[:5]}...")

            # Prepare product information for filtering and display
            product_info = {}
            for idx, product_id in enumerate(self.user_product_matrix.columns):
                product_info[product_id] = {
                    'score': predicted_scores[idx]
                }

            # Add product details if available in product_data
            if 'Product_ID' in self.product_data.columns:
                for product_id in recommended_product_ids:
                    product_row = self.product_data[self.product_data['Product_ID'] == product_id]
                    if not product_row.empty:
                        # Get product name
                        for name_col in ['Product_Name', 'Name', 'Title', 'Description']:
                            if name_col in product_row.columns:
                                product_info[product_id]['name'] = product_row[name_col].iloc[0]
                                break
                        else:
                            product_info[product_id]['name'] = f"Product {product_id}"

                        # Get product price
                        for price_col in ['Price', 'Cost', 'Value']:
                            if price_col in product_row.columns:
                                product_info[product_id]['price'] = product_row[price_col].iloc[0]
                                break
                        else:
                            product_info[product_id]['price'] = 100  # Default price

                        # Get product category
                        for cat_col in ['Category', 'ProductCategory', 'Type', 'Department']:
                            if cat_col in product_row.columns:
                                product_info[product_id]['category'] = product_row[cat_col].iloc[0]
                                break
                        else:
                            product_info[product_id]['category'] = "Uncategorized"

            # Filter recommendations based on price and category
            filtered_recommendations = []
            for product_id in recommended_product_ids:
                info = product_info.get(product_id, {})

                # Skip if we don't have the info needed for filtering
                if 'price' not in info or 'category' not in info:
                    filtered_recommendations.append({
                        'id': product_id,
                        'name': info.get('name', f"Product {product_id}"),
                        'score': info.get('score', 0)
                    })
                    continue

                # Apply price filter
                if price_range and (info['price'] < price_range[0] or info['price'] > price_range[1]):
                    continue

                # Apply category filter
                if category and info['category'] != category:
                    continue

                filtered_recommendations.append({
                    'id': product_id,
                    'name': info.get('name', f"Product {product_id}"),
                    'price': info.get('price', "Unknown"),
                    'category': info.get('category', "Uncategorized"),
                    'score': info.get('score', 0)
                })

            # Store recommendations for this user
            self.recommendations[user_id] = filtered_recommendations

            # Display recommendations
            if not filtered_recommendations:
                print(f"No recommendations available for User_ID {user_id} after filtering.")
            else:
                print(f"Top recommendations for User_ID {user_id}:")
                for i, rec in enumerate(filtered_recommendations[:5], 1):
                    print(f"{i}. {rec.get('name')} (Score: {rec.get('score'):.2f})")
                if len(filtered_recommendations) > 5:
                    print(f"... and {len(filtered_recommendations) - 5} more recommendations")

            return True
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            import traceback
            traceback.print_exc()
            return False

    def create_personalized_message(self, user_id):
        """
        Create a personalized message or offer for the user.
        """
        print(f"Creating personalized message for User_ID: {user_id}...")

        # Check if we have recommendations for this user
        if user_id not in self.recommendations:
            print(f"No recommendations found for User_ID {user_id}. Generating recommendations...")
            if not self.generate_recommendations(user_id):
                return None

        recommendations = self.recommendations[user_id]
        if not recommendations:
            print(f"No recommendations available for User_ID {user_id}.")
            return None

        # Get user information if available
        user_info = "valued customer"
        if 'Customer_ID' in self.user_data.columns:
            user_row = self.user_data[self.user_data['Customer_ID'] == user_id]
            if not user_row.empty:
                # Try to get user's name or other identifying information
                for name_col in ['Name', 'Customer_Name', 'First_Name']:
                    if name_col in user_row.columns:
                        user_info = user_row[name_col].iloc[0]
                        break

        # Create personalized message with top 3 recommendations
        top_recs = recommendations[:3]
        rec_names = [rec.get('name', f"Product {rec.get('id')}") for rec in top_recs]

        message = f"Hello, {user_info}! Based on your preferences, we recommend you check out these products: {', '.join(rec_names)}."

        # Add additional personalization based on available data
        if len(recommendations) > 3:
            message += f" We have {len(recommendations) - 3} more recommendations tailored just for you!"

        print("Message created successfully!")
        return message


def run_recommendation_engine_agent():
    # Paths to datasets
    user_data_path = r"E:\another\multiagent\data\cleaned_customer_data.csv"  # Replace with actual user data path
    product_data_path = r"E:\another\multiagent\data\cleaned_product_data.csv"  # Replace with actual product data path

    # Initialize agent
    agent = RecommendationEngineAgent(user_data_path, product_data_path)

    # Step 1: Fetch data
    if not agent.fetch_data():
        print("Failed to fetch data. Exiting.")
        return

    # Command-line interface
    while True:
        print("\nRecommendation Engine Menu:")
        print("1. Generate Recommendations")
        print("2. Create Personalized Message")
        print("3. Build/Rebuild Recommendation Model")
        print("4. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            try:
                user_id = input("Enter User ID: ")
                # Convert to int if it's a number, otherwise keep as string
                try:
                    user_id = int(user_id)
                except ValueError:
                    pass

                price_min = input("Enter minimum price (or press Enter for no minimum): ")
                price_max = input("Enter maximum price (or press Enter for no maximum): ")

                # Convert price inputs to numbers if provided
                price_range = None
                if price_min and price_max:
                    try:
                        price_min = float(price_min)
                        price_max = float(price_max)
                        if price_min < price_max:
                            price_range = (price_min, price_max)
                    except ValueError:
                        print("Invalid price range. Using no price filter.")

                category = input("Enter product category (optional): ")
                if not category.strip():
                    category = None

                agent.generate_recommendations(user_id, price_range=price_range, category=category)
            except Exception as e:
                print(f"Error processing input: {e}")

        elif choice == "2":
            try:
                user_id = input("Enter User ID: ")
                # Convert to int if it's a number, otherwise keep as string
                try:
                    user_id = int(user_id)
                except ValueError:
                    pass

                message = agent.create_personalized_message(user_id)
                if message:
                    print("Personalized Message:")
                    print(message)
                else:
                    print("No personalized message could be created for this user.")
            except Exception as e:
                print(f"Error processing input: {e}")

        elif choice == "3":
            print("Building/rebuilding recommendation model...")
            agent.build_recommendation_model()

        elif choice == "4":
            print("Exiting Recommendation Engine...")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    run_recommendation_engine_agent()