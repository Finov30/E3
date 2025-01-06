import logging
import os
from configuration.config import Config
from utils import visualization, data_preprocessing, model_training
from utils.data_preprocessing import DataPreprocessor
from utils.model_training import ModelTrainer
from utils.visualization import Visualizer

def setup_logging():
    """Setup logging configuration"""
    os.makedirs(Config.LOGS_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=Config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(Config.LOG_FILE),
            logging.StreamHandler()
        ]
    )

def main():
    # Setup logging
    setup_logging()
    
    try:
        # Initialize components
        preprocessor = DataPreprocessor(max_features=Config.MAX_FEATURES)
        trainer = ModelTrainer()
        visualizer = Visualizer()
        
        # Load and preprocess data
        logging.info("Loading and preprocessing data...")
        df = preprocessor.load_data(Config.DATA_FILE)
        
        # Prepare features
        logging.info("Preparing text features...")
        X, y = preprocessor.prepare_features(df)
        
        # Create sample input for model signature
        sample_input = preprocessor.create_sample_input()
        
        # Split data
        X_train, X_test, y_train, y_test = preprocessor.split_data(
            X, y,
            test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_STATE
        )
        
        # Train and evaluate models
        logging.info("Training and evaluating models...")
        results = trainer.train_and_evaluate(
            X_train, X_test, y_train, y_test,
            sample_input,
            Config.EXPERIMENT_NAME
        )
        
        # Generate and save visualization
        logging.info("Generating model comparison visualization...")
        plt = visualizer.plot_model_comparison(
            results,
            save_path=Config.PLOT_SAVE_PATH
        )
        plt.close()
        
        logging.info("Pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
