import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, max_features=5000, n_components=100):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)

    def load_data(self, file_path):
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Data loaded successfully from {file_path}")
            return df
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def validate_data(self, df):
        try:
            if df.empty:
                raise ValueError("DataFrame is empty")
            required_columns = ['Ticket Description', 'Ticket Priority']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' is missing")
            logging.info("Data validation passed")
        except Exception as e:
            logging.error(f"Data validation error: {str(e)}")
            raise

    def prepare_features(self, df):
        try:
            self.validate_data(df)
            
            # Extract features and target
            X = df['Ticket Description']
            y = df['Ticket Priority']
            
            # Vectorize text data
            X_tfidf = self.vectorizer.fit_transform(X)
            
            # Dimensionality reduction
            X_reduced = self.svd.fit_transform(X_tfidf)
            
            logging.info("Features prepared successfully")
            return X_reduced, y
        except Exception as e:
            logging.error(f"Error preparing features: {str(e)}")
            raise

    def split_data(self, X, y, test_size=0.2, random_state=42):
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            logging.info("Data split into training and testing sets successfully")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error splitting data: {str(e)}")
            raise