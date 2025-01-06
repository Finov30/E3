import logging
import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

class ModelTrainer:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svc': SVC(kernel='rbf', random_state=42)
        }
        
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, sample_input, experiment_name):
        results = {}
        
        # Convert to DataFrame if not already to preserve feature names
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)
        
        mlflow.set_experiment(experiment_name)
        
        for model_name, model in self.models.items():
            with mlflow.start_run(run_name=model_name):
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions on both train and test sets
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # Calculate metrics
                train_score = accuracy_score(y_train, y_train_pred)
                test_score = accuracy_score(y_test, y_test_pred)
                report = classification_report(y_test, y_test_pred)
                
                # Log metrics
                mlflow.log_metric("train_score", train_score)
                mlflow.log_metric("test_score", test_score)
                mlflow.log_param("model_type", model_name)
                
                # Log model
                mlflow.sklearn.log_model(
                    model,
                    model_name,
                    input_example=sample_input
                )
                
                results[model_name] = {
                    'model': model,
                    'train_score': train_score,
                    'test_score': test_score,
                    'report': report
                }
                
                logging.info(f"{model_name} - Train Score: {train_score:.4f}, Test Score: {test_score:.4f}")
                
        return results
