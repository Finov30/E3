import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import json
import os
from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer
import logging

class ModelMonitor:
    def __init__(self):
        self.metrics = {}
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def to_dict(self):
        return {
            "metrics": self.metrics,
            "timestamp": self.timestamp
        }

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Convert date columns
    date_columns = ['Created Date', 'Resolution Date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Handle missing values
    df = df.fillna({
        'Ticket Description': 'No description provided',
        'Ticket Priority': 'Medium',
        'Ticket Type': 'Other'
    })
    
    # Add sentiment analysis
    sia = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df['Ticket Description'].apply(
        lambda x: sia.polarity_scores(str(x))['compound']
    )
    
    return df

def calculate_response_metrics(df):
    metrics = {}
    
    # Calculate average response time
    if 'Created Date' in df.columns and 'Resolution Date' in df.columns:
        df['response_time'] = (df['Resolution Date'] - df['Created Date']).dt.total_seconds() / 3600
        metrics['avg_response_time_hours'] = df['response_time'].mean()
    
    # Calculate satisfaction metrics if available
    if 'Customer Satisfaction' in df.columns:
        metrics['avg_satisfaction'] = df['Customer Satisfaction'].mean()
    
    # Calculate sentiment metrics
    if 'sentiment_score' in df.columns:
        metrics['avg_sentiment'] = df['sentiment_score'].mean()
    
    return metrics

def prepare_text_features(df):
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    X = vectorizer.fit_transform(df['Ticket Description'].astype(str))
    return X, vectorizer

def train_ticket_classifier(X, y):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X, y)
    return model

def save_model_and_artifacts(model, vectorizer, monitor, version="1.0"):
    # Create directory if it doesn't exist
    os.makedirs('./models', exist_ok=True)
    
    # Save the model and vectorizer using joblib
    model_path = os.path.join('./models', f'ticket_classifier_v{version}.joblib')
    vectorizer_path = os.path.join('./models', f'vectorizer_v{version}.joblib')
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    # Save configuration and metrics
    config = {
        'version': version,
        'model_path': model_path,
        'vectorizer_path': vectorizer_path,
        'model_params': model.get_params(),
        'vectorizer_params': vectorizer.get_params(),
        'monitor': monitor.to_dict(),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    config_path = os.path.join('./models', f'config_v{version}.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4, default=str)
    
    logging.info(f"Model and artifacts saved successfully to {os.path.abspath('./models')}")
