import logging
import os
from utils import (
    train_and_evaluate_models,
    visualize_model_comparison
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import mlflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    try:
        # Load and preprocess data
        logging.info("Loading and preprocessing data...")
        df = pd.read_csv('./customer_support_tickets.csv')
        
        # Prepare features
        logging.info("Preparing text features...")
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(df['Ticket Description'].astype(str))
        y = df['Ticket Priority']
        
        # Create sample input for model signature
        sample_input = pd.DataFrame(
            vectorizer.transform(['Sample ticket description']).toarray(),
            columns=vectorizer.get_feature_names_out()
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train and evaluate models
        logging.info("Training and evaluating models...")
        results = train_and_evaluate_models(
            X_train, X_test, y_train, y_test,
            sample_input,
            experiment_name="Customer_Support_Classification"
        )
        
        # Generate and save visualization
        logging.info("Generating model comparison visualization...")
        plt = visualize_model_comparison(results, save_path='./model_comparison.png')
        plt.close()  # Ensure the plot is closed
        
        logging.info("Pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
