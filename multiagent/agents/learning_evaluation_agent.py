import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import joblib


class LearningEvaluationAgent:
    def __init__(self, model_path='evaluation_model.pkl', scaler_path='evaluation_scaler.pkl'):
        """
        Initialize the Learning Evaluation Agent.
        """
        self.model = None
        self.scaler = None
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.evaluation_results = {}

    def preprocess_data(self, data, label_column):
        """
        Preprocess the dataset to handle imbalanced classes, scale features, and split into train/test sets.
        """
        print("Preprocessing data...")
        X = data.drop(columns=[label_column])
        y = data[label_column]

        # Handle imbalanced classes using SMOTE
        smote = SMOTE(k_neighbors=3, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        self.scaler = scaler

        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        """
        Train and optimize a Random Forest Classifier using Grid Search.
        """
        print("Training and optimizing the model...")
        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_
        print(f"Best Model Parameters: {grid_search.best_params_}")

        # Save the model
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        print(f"Model saved to {self.model_path}")

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model and generate key metrics such as accuracy, precision, recall, and confusion matrix.
        """
        print("Evaluating model performance...")
        y_pred = self.model.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Store results for further analysis
        self.evaluation_results = {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": conf_matrix
        }

        # Print metrics
        print(f"Model Accuracy: {accuracy * 100:.2f}%")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(conf_matrix)
        return self.evaluation_results

    def visualize_results(self):
        """
        Visualize model evaluation results using Streamlit.
        """
        if self.evaluation_results:
            st.write("### Model Evaluation Results")

            # Accuracy Visualization
            accuracy = self.evaluation_results["accuracy"]
            st.write(f"**Accuracy:** {accuracy * 100:.2f}%")

            # Confusion Matrix Visualization
            conf_matrix = self.evaluation_results["confusion_matrix"]
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues")
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(plt)

            # Classification Report Visualization
            report = pd.DataFrame(self.evaluation_results["classification_report"]).transpose()
            st.write("### Classification Report")
            st.dataframe(report)
        else:
            st.warning("No evaluation results available. Please evaluate the model first.")

    def load_model(self):
        """
        Load a previously trained model from disk.
        """
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            print(f"Model loaded from {self.model_path}")
            return True
        except FileNotFoundError:
            print(f"Model file not found at {self.model_path}")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Streamlit Dashboard
    st.title("Learning Evaluation Agent Dashboard")

    # Example dataset (replace with your actual dataset)
    data = pd.DataFrame({
        'Browsing_History': [3, 5, 1, 0, 2, 4, 5, 2, 1, 3],
        'Purchase_History': [0, 1, 0, 1, 1, 0, 1, 0, 0, 1],
        'Customer_Segment': [0, 1, 0, 1, 1, 0, 1, 0, 0, 1]
    })

    agent = LearningEvaluationAgent()

    # Preprocess the data
    X_train, X_test, y_train, y_test = agent.preprocess_data(data, label_column='Customer_Segment')

    # Train and evaluate the model
    agent.train_model(X_train, y_train)
    evaluation_results = agent.evaluate_model(X_test, y_test)

    # Visualize results
    agent.visualize_results()