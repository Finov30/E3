import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import os
import json

def train_and_evaluate_models(X_train, X_test, y_train, y_test, sample_input, experiment_name="Customer_Support_Classification"):
    """Train and evaluate models with proper MLflow tracking and model signatures"""
    
    # Ensure the experiment exists
    mlflow.set_experiment(experiment_name)
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(kernel='linear', random_state=42, probability=True)
    }
    
    results = {}
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            # Train model
            model.fit(X_train, y_train)
            
            # Log metrics
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            mlflow.log_metric("train_accuracy", train_score)
            mlflow.log_metric("test_accuracy", test_score)
            
            # Log model with signature
            signature = mlflow.models.infer_signature(
                sample_input,
                model.predict(sample_input)
            )
            
            mlflow.sklearn.log_model(
                model,
                name,
                signature=signature,
                input_example=sample_input
            )
            
            results[name] = {
                'train_score': train_score,
                'test_score': test_score,
                'model': model
            }
            
            # Log parameters
            mlflow.log_params(model.get_params())
    
    return results

def visualize_model_comparison(results, save_path=None):
    """Create and save model comparison visualization with non-blocking display"""
    plt.figure(figsize=(10, 6))
    
    models = list(results.keys())
    train_scores = [results[m]['train_score'] for m in models]
    test_scores = [results[m]['test_score'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, train_scores, width, label='Train Score')
    plt.bar(x + width/2, test_scores, width, label='Test Score')
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()  # Close the figure to prevent display blocking
    
    return plt
