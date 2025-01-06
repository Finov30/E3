import os

class Config:
    # Data paths
    DATA_FILE = "src/data/customer_support_tickets.csv"
    
    # Model parameters
    MAX_FEATURES = 1000
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # MLflow settings
    EXPERIMENT_NAME = "Customer_Support_Classification"
    
    # Logging configuration
    LOGS_DIR = "logs"
    LOG_FILE = os.path.join(LOGS_DIR, "pipeline.log")
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    
    # Visualization
    PLOT_SAVE_PATH = "src/visualizations/model_comparison.png"
