import os


class Config:
    MLFLOW_TRACKING_URI = "http://localhost:4000"
    EXPERIMENT_NAME = "ticket_classification"
    DATA_PATH = "project\src\data\customer_support_tickets.csv"
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    MAX_FEATURES = 5000
    N_COMPONENTS = 100