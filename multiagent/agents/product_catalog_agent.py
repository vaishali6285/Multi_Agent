import pandas as pd
import numpy as np
import sqlite3
import os
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import joblib


class ProductCatalogAgent:
    def __init__(self, db_path, embeddings_path='product_embeddings.pkl'):
        """
        Initialize the Product Catalog Agent with database connection and embedding storage.

        Args:
            db_path (str): Path to the SQLite database
            embeddings_path (str): Path to save product embeddings
        """
        self.db_path = db_path
        self.embeddings_path = embeddings_path
        self.product_embeddings = {}
        self.similarity_matrix = None
        self.product_ids = []  # Store product IDs for similarity lookup

        # Ensure database directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')

    def connect_to_db(self):
        """Establish connection to the SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            return conn
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            return None

    def initialize_product_table(self):
        """Create the product catalog table if it doesn't exist."""
        conn = self.connect_to_db()
        if conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ProductCatalog (
                        product_id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        category TEXT,
                        description TEXT,
                        price REAL,
                        brand TEXT,
                        attributes TEXT,
                        inventory_count INTEGER,
                        view_count INTEGER DEFAULT 0,
                        purchase_count INTEGER DEFAULT 0
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ProductRelationships (
                        product_id INTEGER,
                        related_product_id INTEGER,
                        relationship_type TEXT,
                        strength REAL,
                        PRIMARY KEY (product_id, related_product_id),
                        FOREIGN KEY (product_id) REFERENCES ProductCatalog (product_id),
                        FOREIGN KEY (related_product_id) REFERENCES ProductCatalog (product_id)
                    )
                """)

                conn.commit()
                print("ProductCatalog and ProductRelationships tables created successfully!")
            except sqlite3.Error as e:
                print(f"Table creation error: {e}")
            finally:
                conn.close()
        else:
            print("Failed to connect to the database.")

    def add_sample_products(self):
        """Add sample products to the database if it's empty."""
        conn = self.connect_to_db()
        if conn:
            cursor = conn.cursor()
            try:
                # Check if products already exist
                cursor.execute("SELECT COUNT(*) FROM ProductCatalog")
                count = cursor.fetchone()[0]

                if count == 0:
                    # Sample products
                    products = [
                        (1, "iPhone 15 Pro", "Smartphones", "Latest Apple smartphone with A17 Pro chip and 48MP camera",
                         999.99, "Apple", "color:space black,storage:256GB", 50, 120, 25),
                        (2, "Samsung Galaxy S24", "Smartphones", "Flagship Android smartphone with AI features", 899.99,
                         "Samsung", "color:phantom black,storage:128GB", 45, 100, 20),
                        (3, "MacBook Air M3", "Laptops", "Ultra-thin laptop with Apple M3 chip", 1299.99, "Apple",
                         "color:silver,storage:512GB", 30, 80, 15),
                        (4, "Dell XPS 15", "Laptops", "Premium Windows laptop with OLED display", 1799.99, "Dell",
                         "color:platinum silver,storage:1TB", 25, 60, 12),
                        (5, "Sony WH-1000XM5", "Headphones", "Premium noise-cancelling headphones", 349.99, "Sony",
                         "color:black,type:over-ear", 60, 150, 35),
                        (6, "iPad Pro 12.9", "Tablets", "Large tablet with M2 chip and Liquid Retina XDR display",
                         1099.99, "Apple", "color:space gray,storage:256GB", 40, 90, 18),
                        (7, "Apple Watch Series 9", "Smartwatches", "Health and fitness smartwatch", 399.99, "Apple",
                         "color:midnight,size:45mm", 35, 85, 22),
                        (8, "Samsung 65\" QLED TV", "TVs", "4K QLED TV with smart features", 1299.99, "Samsung",
                         "size:65inch,resolution:4K", 20, 40, 8),
                        (9, "Bose QuietComfort Earbuds", "Headphones", "Noise-cancelling wireless earbuds", 249.99,
                         "Bose", "color:white,type:in-ear", 55, 130, 28),
                        (10, "Logitech MX Master 3S", "Computer Accessories", "Ergonomic wireless mouse", 99.99,
                         "Logitech", "color:graphite,connectivity:bluetooth", 70, 110, 30)
                    ]

                    cursor.executemany("""
                        INSERT INTO ProductCatalog (product_id, name, category, description, price, brand, attributes, 
                                                  inventory_count, view_count, purchase_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, products)

                    # Sample relationships (complementary, substitute, etc.)
                    relationships = [
                        (1, 7, "complementary", 0.85),  # iPhone and Apple Watch
                        (1, 9, "complementary", 0.75),  # iPhone and Earbuds
                        (3, 10, "complementary", 0.8),  # MacBook and Mouse
                        (1, 2, "substitute", 0.9),  # iPhone and Galaxy
                        (3, 4, "substitute", 0.85),  # MacBook and XPS
                        (5, 9, "substitute", 0.7),  # Sony headphones and Bose earbuds
                        (1, 6, "upsell", 0.6),  # iPhone and iPad
                        (2, 8, "upsell", 0.5)  # Galaxy and Samsung TV
                    ]

                    cursor.executemany("""
                        INSERT INTO ProductRelationships (product_id, related_product_id, relationship_type, strength)
                        VALUES (?, ?, ?, ?)
                    """, relationships)

                    conn.commit()
                    print("Sample products and relationships added successfully!")
                else:
                    print("Products already exist in the database.")
            except sqlite3.Error as e:
                print(f"Error adding sample products: {e}")
            finally:
                conn.close()
        else:
            print("Failed to connect to the database.")

    def generate_product_embeddings(self):
        """
        Generate embeddings for products based on their attributes and descriptions.
        """
        conn = self.connect_to_db()
        if conn:
            try:
                df = pd.read_sql_query("""
                    SELECT product_id, name, category, description, brand, attributes 
                    FROM ProductCatalog
                """, conn)

                # Check if any products were found
                if df.empty:
                    print("No products found in database. Please add products first.")
                    return False

                # Create feature text for each product
                df['features'] = df['name'] + ' ' + df['category'] + ' ' + df['description'] + ' ' + df['brand'] + ' ' + \
                                 df['attributes'].fillna('')  # Handle NULL attributes

                # Simple text processing
                stop_words = set(stopwords.words('english'))

                # Generate simple TF-IDF-like vectors (simplified for demonstration)
                embeddings = {}
                all_tokens = set()

                # Get all unique tokens - with better error handling
                for idx, row in df.iterrows():
                    try:
                        tokens = word_tokenize(str(row['features']).lower())
                        tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
                        all_tokens.update(tokens)
                    except Exception as e:
                        print(f"Error tokenizing row {idx}: {e}")
                        continue

                # Create a vocabulary mapping
                vocab = {token: i for i, token in enumerate(all_tokens)}

                # Check if we have any tokens
                if not vocab:
                    print("No valid tokens found in product descriptions.")
                    return False

                # Generate embeddings
                product_ids = []
                embedding_vectors = []

                for idx, row in df.iterrows():
                    try:
                        tokens = word_tokenize(str(row['features']).lower())
                        tokens = [t for t in tokens if t.isalnum() and t not in stop_words]

                        # Create a simple count-based vector
                        vector = np.zeros(len(vocab))
                        for token in tokens:
                            if token in vocab:
                                vector[vocab[token]] += 1

                        # Normalize the vector
                        norm = np.linalg.norm(vector)
                        if norm > 0:
                            vector = vector / norm

                            # Store the embedding and product ID
                            product_id = row['product_id']
                            embeddings[product_id] = vector
                            product_ids.append(product_id)
                            embedding_vectors.append(vector)

                    except Exception as e:
                        print(f"Error generating embedding for row {idx}: {e}")
                        continue

                # Check if we have any embeddings
                if not embedding_vectors:
                    print("No embeddings were generated. Check the product data.")
                    return False

                # Store the product IDs and embeddings
                self.product_embeddings = embeddings
                self.product_ids = product_ids

                # Calculate similarity matrix from the embedding vectors (not from an empty array)
                embedding_matrix = np.array(embedding_vectors)
                self.similarity_matrix = cosine_similarity(embedding_matrix)

                # Ensure directory exists for embeddings
                embedding_dir = os.path.dirname(self.embeddings_path)
                if embedding_dir and not os.path.exists(embedding_dir):
                    os.makedirs(embedding_dir, exist_ok=True)

                # Save embeddings and product IDs
                joblib.dump({
                    'embeddings': self.product_embeddings,
                    'product_ids': product_ids,
                    'similarity_matrix': self.similarity_matrix
                }, self.embeddings_path)

                print(f"Product embeddings generated and saved to {self.embeddings_path}")
                return True
            except Exception as e:
                print(f"Error generating product embeddings: {e}")
                import traceback
                traceback.print_exc()  # Print full traceback for debugging
                return False
            finally:
                conn.close()
        else:
            print("Failed to connect to the database.")
            return False

    def load_embeddings(self):
        """Load pre-computed embeddings from disk."""
        try:
            if os.path.exists(self.embeddings_path):
                data = joblib.load(self.embeddings_path)
                self.product_embeddings = data['embeddings']
                self.product_ids = data['product_ids']  # Load product IDs into class instance
                self.similarity_matrix = data['similarity_matrix']
                print(f"Embeddings loaded from {self.embeddings_path}")
                return True
            else:
                print(f"Embeddings file not found at {self.embeddings_path}")
                return False
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return False

    def get_similar_products(self, product_id, n=5):
        """
        Find similar products based on embeddings.

        Args:
            product_id (int): The product ID to find similar products for
            n (int): Number of similar products to return

        Returns:
            list: List of tuples (product_id, similarity_score)
        """
        # Load or generate embeddings if they don't exist
        if not self.product_embeddings or not self.similarity_matrix or not self.product_ids:
            success = self.load_embeddings()
            if not success:
                success = self.generate_product_embeddings()
                if not success:
                    print("Could not generate or load embeddings")
                    return []

        # Check if the product ID exists in our data
        if product_id not in self.product_ids:
            print(f"Product ID {product_id} not found in embeddings.")
            return []

        # Get the index of the product ID
        idx = self.product_ids.index(product_id)

        # Get similarities from the matrix
        similarities = self.similarity_matrix[idx]

        # Make sure we have enough products to return n similar ones
        n = min(n, len(self.product_ids) - 1)
        if n <= 0:
            return []

        # Get top N similar products (excluding self)
        similar_indices = similarities.argsort()[::-1][1:n + 1]
        return [(self.product_ids[i], similarities[i]) for i in similar_indices]

    def get_related_products(self, product_id, relationship_type=None):
        """
        Get products related to the given product ID.

        Args:
            product_id (int): The product ID to find related products for
            relationship_type (str, optional): Type of relationship (complementary, substitute, etc.)

        Returns:
            list: List of related products with relationship info
        """
        conn = self.connect_to_db()
        if conn:
            try:
                query = """
                    SELECT r.related_product_id, p.name, r.relationship_type, r.strength
                    FROM ProductRelationships r
                    JOIN ProductCatalog p ON r.related_product_id = p.product_id
                    WHERE r.product_id = ?
                """

                params = [product_id]

                if relationship_type:
                    query += " AND r.relationship_type = ?"
                    params.append(relationship_type)

                query += " ORDER BY r.strength DESC"

                df = pd.read_sql_query(query, conn, params=params)
                return df.to_dict('records')
            except Exception as e:
                print(f"Error getting related products: {e}")
                return []
            finally:
                conn.close()
        else:
            print("Failed to connect to the database.")
            return []

    def update_product_metrics(self, product_id, view_increment=0, purchase_increment=0):
        """
        Update product view and purchase counts.

        Args:
            product_id (int): The product ID to update
            view_increment (int): Number of views to add
            purchase_increment (int): Number of purchases to add
        """
        conn = self.connect_to_db()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE ProductCatalog
                    SET view_count = view_count + ?,
                        purchase_count = purchase_count + ?
                    WHERE product_id = ?
                """, (view_increment, purchase_increment, product_id))
                conn.commit()
                print(f"Product {product_id} metrics updated successfully.")
            except sqlite3.Error as e:
                print(f"Error updating product metrics: {e}")
            finally:
                conn.close()
        else:
            print("Failed to connect to the database.")

    def search_products(self, keyword):
        """
        Search for products based on keyword in name, description, or category.

        Args:
            keyword (str): Search keyword

        Returns:
            list: List of matching products
        """
        conn = self.connect_to_db()
        if conn:
            try:
                query = """
                    SELECT product_id, name, category, description, price, brand
                    FROM ProductCatalog
                    WHERE name LIKE ? OR category LIKE ? OR description LIKE ? OR brand LIKE ?
                    ORDER BY view_count DESC
                """

                search_pattern = f"%{keyword}%"
                df = pd.read_sql_query(query, conn,
                                       params=[search_pattern, search_pattern, search_pattern, search_pattern])
                return df.to_dict('records')
            except Exception as e:
                print(f"Error searching products: {e}")
                return []
            finally:
                conn.close()
        else:
            print("Failed to connect to the database.")
            return []


# Example usage
if __name__ == "__main__":
    db_path = os.path.join(os.path.dirname(__file__), "db", "multi_agent_system.db")
    agent = ProductCatalogAgent(db_path)

    # Initialize database
    agent.initialize_product_table()
    agent.add_sample_products()

    # Generate product embeddings
    agent.generate_product_embeddings()

    # Test search functionality
    print("\nSearching for 'Apple' products:")
    apple_products = agent.search_products("Apple")
    for product in apple_products:
        print(f"{product['name']} - {product['category']} - ${product['price']}")

    # Test similar products
    print("\nProducts similar to iPhone 15 Pro:")
    similar_products = agent.get_similar_products(1, n=3)
    for pid, score in similar_products:
        print(f"Product ID: {pid}, Similarity Score: {score:.2f}")

    # Test related products
    print("\nComplementary products for iPhone:")
    related_products = agent.get_related_products(1, "complementary")
    for product in related_products:
        print(f"{product['name']} - Relationship: {product['relationship_type']} - Strength: {product['strength']}")