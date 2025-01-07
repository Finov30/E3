import logging
import mlflow
import time
import requests
from utils import visualization, data_preprocessing, model_training
from utils.data_preprocessing import DataPreprocessor
from utils.model_training import ModelTrainer
from utils.visualization import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def start_mlflow_server():
    try:
        requests.get('http://localhost:4000')
        logging.info("MLflow server is already running")
    except:
        logging.info("Starting MLflow server...")
        import subprocess
        subprocess.Popen(["mlflow", "ui", "--host", "0.0.0.0", "--port", "4000"])
        time.sleep(5)
        logging.info("MLflow server started")

def main():
    try:
        # Start MLflow server
        start_mlflow_server()
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri('http://localhost:4000')
        
        # Initialize components
        preprocessor = DataPreprocessor(max_features=5000, n_components=100)
        model_trainer = ModelTrainer()
        visualizer = Visualizer()
        
        # Load and preprocess data
        logging.info("Loading and preprocessing data...")
        df = preprocessor.load_data('project\src\data\customer_support_tickets.csv')
        
        # Prepare features
        X, y = preprocessor.prepare_features(df)
        
        # Split the data
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
        
        # Set experiment name
        experiment_name = "ticket_classification"
        mlflow.set_experiment(experiment_name)
        
        # Train and evaluate models
        logging.info("Training and evaluating models...")
        results = model_trainer.train_and_evaluate(
            X_train, X_test, y_train, y_test,
            experiment_name
        )
        
        # Generate visualizations
        logging.info("Generating visualizations...")
        comparison_plot = visualizer.plot_model_comparison(results)
        comparison_plot.savefig('model_comparison.png')
        
        logging.info("Pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in pipeline: {str(e)}")
        raise
    finally:
        logging.info("Pipeline execution completed")

if __name__ == "__main__":
    main()