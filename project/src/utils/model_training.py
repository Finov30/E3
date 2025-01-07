
import logging
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

class ModelTrainer:
    def __init__(self):
        # Simple model configurations with fixed parameters
        self.base_models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            )
        }
        self.scaler = StandardScaler()
        self.best_models = {}

    def train_and_evaluate(self, X_train, X_test, y_train, y_test, experiment_name):
        logging.info("Starting model training and evaluation")
        mlflow.set_experiment(experiment_name)
        results = {}

        # Scale features
        logging.info("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        logging.info("Features scaled successfully")

        # Train and evaluate base models
        for model_name, model in self.base_models.items():
            logging.info(f"Training model: {model_name}")
            
            try:
                with mlflow.start_run(run_name=model_name) as run:
                    # Train model
                    if model_name == 'logistic_regression':
                        model.fit(X_train_scaled, y_train)
                        y_train_pred = model.predict(X_train_scaled)
                        y_test_pred = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_train_pred = model.predict(X_train)
                        y_test_pred = model.predict(X_test)

                    # Calculate metrics
                    train_score = accuracy_score(y_train, y_train_pred)
                    test_score = accuracy_score(y_test, y_test_pred)

                    # Log metrics
                    mlflow.log_metric("train_accuracy", train_score)
                    mlflow.log_metric("test_accuracy", test_score)

                    # Log model
                    mlflow.sklearn.log_model(
                        model,
                        f"model_{model_name}"
                    )
                    
                    # Store results
                    self.best_models[model_name] = model
                    results[model_name] = {
                        'model': model,
                        'train_score': train_score,
                        'test_score': test_score,
                        'predictions': y_test_pred
                    }
                    
                    logging.info(f"Completed training for {model_name}")
                    logging.info(f"Train accuracy: {train_score:.4f}")
                    logging.info(f"Test accuracy: {test_score:.4f}")

            except Exception as e:
                logging.error(f"Error during training {model_name}: {str(e)}")
                continue

        logging.info("Model training and evaluation completed")
        return results
