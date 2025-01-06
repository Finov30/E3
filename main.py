
import pandas as pd
import logging
from datetime import datetime
import os
import nltk

# Download required NLTK data at startup
try:
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    logging.error(f"Failed to download NLTK resources: {str(e)}")

from utils import (
    load_and_preprocess_data,
    calculate_response_metrics,
    prepare_text_features,
    train_ticket_classifier,
    save_model_and_artifacts,
    ModelMonitor
)

def main():
    try:
        # Create necessary directories
        os.makedirs('./visualizations', exist_ok=True)
        os.makedirs('./models', exist_ok=True)
        
        # Load and preprocess data
        logging.info("Loading data...")
        df = load_and_preprocess_data('./customer_support_tickets.csv')
        logging.info(f"Data loaded successfully. Shape: {df.shape}")

        # Calculate performance metrics
        logging.info("Calculating performance metrics...")
        metrics = calculate_response_metrics(df)
        logging.info("Metrics calculated:")
        for metric, value in metrics.items():
            logging.info(f"{metric}: {value}")

        # Prepare features for the model
        logging.info("Preparing features...")
        X, vectorizer = prepare_text_features(df)
        
        # Encode target variable
        logging.info("Encoding target variable...")
        y = df['Ticket Priority']
        
        # Train the classifier
        logging.info("Training the classifier...")
        model = train_ticket_classifier(X, y)
        
        # Save the model and artifacts
        logging.info("Saving the model and artifacts...")
        monitor = ModelMonitor()
        save_model_and_artifacts(model, vectorizer, monitor, version="1.0")
        logging.info("Model and artifacts saved successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'support_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    main()
