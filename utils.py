import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

file_path = "C:\\Users\\samua\\Desktop\\Project E2 V4\\customer_support_tickets.csv"

def load_and_preprocess_data(file_path):
    """
    Charge et prétraite les données des tickets support
    """
    try:
        df = pd.read_csv(file_path)
        
        # Conversion des dates
        date_columns = ['Date of Purchase', 'First Response Time', 'Time to Resolution']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Gestion des valeurs manquantes
        df['Ticket Description'] = df['Ticket Description'].fillna('No description provided')
        df['Customer Age'] = df['Customer Age'].fillna(df['Customer Age'].median())
        
        return df
    except Exception as e:
        logging.error(f"Erreur lors du chargement des données: {str(e)}")
        raise

def calculate_response_metrics(df):
    """
    Calcule les métriques de performance du support client
    """
    metrics = {
        'avg_response_time': None,
        'avg_resolution_time': None,
        'satisfaction_score': None
    }
    
    try:
        # Temps moyen de première réponse
        if 'First Response Time' in df.columns:
            df['response_time'] = (df['First Response Time'] - pd.to_datetime(df['Date of Purchase'])).dt.total_seconds() / 3600
            metrics['avg_response_time'] = df['response_time'].mean()

        # Temps moyen de résolution
        if 'Time to Resolution' in df.columns and 'First Response Time' in df.columns:
            df['resolution_time'] = (df['Time to Resolution'] - df['First Response Time']).dt.total_seconds() / 3600
            metrics['avg_resolution_time'] = df['resolution_time'].mean()

        # Score moyen de satisfaction
        if 'Customer Satisfaction Rating' in df.columns:
            metrics['satisfaction_score'] = df['Customer Satisfaction Rating'].mean()

        return metrics
    except Exception as e:
        logging.error(f"Erreur lors du calcul des métriques: {str(e)}")
        return metrics

def create_visualizations(df, output_dir='./visualizations'):
    """
    Crée des visualisations pour l'analyse des tickets
    """
    try:
        # Distribution des types de tickets
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='Ticket Type')
        plt.xticks(rotation=45)
        plt.title('Distribution des Types de Tickets')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ticket_types_distribution.png')
        plt.close()

        # Distribution des priorités
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x='Ticket Priority')
        plt.title('Distribution des Priorités des Tickets')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ticket_priority_distribution.png')
        plt.close()

        # Satisfaction client moyenne par type de ticket
        plt.figure(figsize=(10, 6))
        df.groupby('Ticket Type')['Customer Satisfaction Rating'].mean().plot(kind='bar')
        plt.title('Satisfaction Client Moyenne par Type de Ticket')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/satisfaction_by_ticket_type.png')
        plt.close()

    except Exception as e:
        logging.error(f"Erreur lors de la création des visualisations: {str(e)}")

def prepare_text_features(df):
    """
    Prépare les caractéristiques textuelles pour l'analyse
    """
    try:
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        text_features = vectorizer.fit_transform(df['Ticket Description'])
        
        return text_features, vectorizer
    except Exception as e:
        logging.error(f"Erreur lors de la préparation des caractéristiques textuelles: {str(e)}")
        raise

def train_ticket_classifier(X, y):
    """
    Entraîne un classifieur pour la catégorisation des tickets
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        
        return clf, classification_report(y_test, y_pred), confusion_matrix(y_test, y_pred)
    except Exception as e:
        logging.error(f"Erreur lors de l'entraînement du classifieur: {str(e)}")
        raise

def save_model_and_artifacts(model, vectorizer, label_encoder, output_dir='./models'):
    """
    Sauvegarde le modèle et les artefacts associés
    """
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Sauvegarde du modèle
        joblib.dump(model, f'{output_dir}/ticket_classifier_{timestamp}.joblib')
        
        # Sauvegarde du vectorizer
        joblib.dump(vectorizer, f'{output_dir}/vectorizer_{timestamp}.joblib')
        
        # Sauvegarde du label encoder
        joblib.dump(label_encoder, f'{output_dir}/label_encoder_{timestamp}.joblib')
        
        logging.info(f"Modèle et artefacts sauvegardés avec succès dans {output_dir}")
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde du modèle: {str(e)}")
        raise