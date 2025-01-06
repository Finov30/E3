
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

class DataPreprocessor:
    def __init__(self, max_features=1000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        
    def load_data(self, file_path):
        """Load data from CSV file"""
        return pd.read_csv(file_path)
        
    def prepare_features(self, df):
        """Prepare text features using TF-IDF"""
        # Use the correct column for text data
        X = self.vectorizer.fit_transform(df['Ticket Description'].fillna(""))
        
        # Convert to DataFrame with feature names
        feature_names = self.vectorizer.get_feature_names_out()
        X = pd.DataFrame(X.toarray(), columns=feature_names)
        
        y = df['Ticket Type']
        return X, y
    
    def create_sample_input(self):
        """Create a sample input for model signature"""
        feature_names = self.vectorizer.get_feature_names_out()
        sample = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
        return sample
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
