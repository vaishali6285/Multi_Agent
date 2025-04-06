#!/usr/bin/env python3
import os
import pandas as pd

# Import all agents
from agents.db.customer_intent_agent import CustomerIntentAgent
from db.customer_profile_agent import CustomerProfileAgent
from db.learning_evaluation_agent import LearningEvaluationAgent
from db.personalization_experience_agent import PersonalizationExperienceAgent
from db.product_catalog_agent import ProductCatalogAgent
from db.recommendation_engine_agent import RecommendationEngineAgent


class RecommendationSystem:
    """
    Main class that orchestrates all recommendation system agents.
    """

    def __init__(self, data_path="db"):
        self.data_path = data_path

        # Initialize data sources
        self.customer_data = pd.read_csv(os.path.join(data_path, "cleared_customer_data.csv"))
        self.item_mapping = pd.read_csv(os.path.join(data_path, "item_mapping.csv"))

        # Initialize agents
        self.customer_profile_agent = CustomerProfileAgent(self.customer_data)
        self.product_catalog_agent = ProductCatalogAgent(self.item_mapping)
        self.customer_intent_agent = CustomerIntentAgent()
        self.recommendation_engine = RecommendationEngineAgent(
            model_path=os.path.join(data_path, "evaluation_model.pkl"),
            scaler_path=os.path.join(data_path, "evaluation_scaler.pkl")
        )
        self.personalization_agent = PersonalizationExperienceAgent()
        self.evaluation_agent = LearningEvaluationAgent()

        print("Recommendation system initialized with all agents.")

    def process_customer_recommendation(self, customer_id, context=None):
        """
        Process a complete recommendation flow for a customer.

        Args:
            customer_id: The ID of the customer
            context: Optional contextual information about the current session

        Returns:
            Personalized recommendations for the customer
        """
        print(f"Processing recommendations for customer {customer_id}")

        # Step 1: Get customer profile
        customer_profile = self.customer_profile_agent.get_customer_profile(customer_id)
        print(f"Retrieved customer profile: {customer_profile}")

        # Step 2: Analyze customer intent
        customer_intent = self.customer_intent_agent.analyze_intent(customer_id, context)
        print(f"Analyzed customer intent: {customer_intent}")

        # Step 3: Get available product catalog
        available_products = self.product_catalog_agent.get_available_products(
            categories=customer_profile.get('preferred_categories', [])
        )
        print(f"Retrieved {len(available_products)} relevant products")

        # Step 4: Generate recommendations
        recommendations = self.recommendation_engine.generate_recommendations(
            customer_profile=customer_profile,
            customer_intent=customer_intent,
            available_products=available_products
        )
        print(f"Generated {len(recommendations)} initial recommendations")

        # Step 5: Personalize the recommendations
        personalized_recommendations = self.personalization_agent.personalize_experience(
            customer_id=customer_id,
            recommendations=recommendations,
            context=context
        )
        print(f"Personalized recommendations: {personalized_recommendations}")

        # Step 6: Log for evaluation
        self.evaluation_agent.log_recommendation_event(
            customer_id=customer_id,
            recommendations=personalized_recommendations,
            context=context
        )

        return personalized_recommendations

    def train_models(self):
        """Train or update all models in the system"""
        print("Training recommendation models...")
        self.recommendation_engine.train_model()
        self.customer_intent_agent.train_model()
        print("Model training complete")

    def evaluate_system_performance(self, time_period=None):
        """Evaluate overall system performance"""
        print("Evaluating system performance...")
        metrics = self.evaluation_agent.evaluate_performance(time_period)
        print(f"Performance metrics: {metrics}")
        return metrics


def main():
    """Main entry point demonstrating the system capabilities"""
    print("Starting Recommendation System Demo")

    # Initialize the system
    rec_system = RecommendationSystem()

    # Example: Process recommendations for a customer
    customer_id = "CUST12345"
    context = {
        "session_id": "SESSION98765",
        "current_page": "product_category_electronics",
        "time_on_site": 120,  # seconds
        "device": "mobile",
        "location": "New York"
    }

    # Get personalized recommendations
    recommendations = rec_system.process_customer_recommendation(
        customer_id=customer_id,
        context=context
    )

    # Example: Evaluate system performance
    performance = rec_system.evaluate_system_performance(time_period="last_7_days")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()