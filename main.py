import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
from utils import (
    load_and_preprocess_data,
    calculate_response_metrics,
    create_visualizations,
    prepare_text_features,
    train_ticket_classifier,
    save_model_and_artifacts
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'support_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def main():
    try:
        # Création des répertoires nécessaires
        os.makedirs('./visualizations', exist_ok=True)
        os.makedirs('./models', exist_ok=True)
        
        # Chargement et prétraitement des données
        logging.info("Chargement des données...")
        df = load_and_preprocess_data('./customer_support_tickets.csv')
        logging.info(f"Données chargées avec succès. Shape: {df.shape}")

        # Calcul des métriques de performance
        logging.info("Calcul des métriques de performance...")
        metrics = calculate_response_metrics(df)
        logging.info("Métriques calculées:")
        for metric, value in metrics.items():
            logging.info(f"{metric}: {value}")

        # Création des visualisations
        logging.info("Création des visualisations...")
        create_visualizations(df)
        logging.info("Visualisations créées avec succès")

        # Préparation des caractéristiques pour le modèle
        logging.info("Préparation des caractéristiques...")
        X, vectorizer = prepare_text_features(df)
        y = df['Ticket Type']
        
        # Entraînement du modèle
        logging.info("Entraînement du modèle de classification...")
        model, classification_rep, conf_matrix = train_ticket_classifier(X, y)
        logging.info("Rapport de classification:")
        logging.info(classification_rep)

        # Sauvegarde du modèle et des artefacts
        logging.info("Sauvegarde du modèle et des artefacts...")
        save_model_and_artifacts(model, vectorizer, None)
        
        logging.info("Analyse terminée avec succès!")
        
        # Affichage des résultats principaux
        print("\
Résultats principaux:")
        print("=====================")
        print(f"Nombre total de tickets: {len(df)}")
        print(f"Types de tickets uniques: {df['Ticket Type'].nunique()}")
        print(f"Satisfaction client moyenne: {metrics['satisfaction_score']:.2f}")
        print(f"Temps moyen de résolution (heures): {metrics['avg_resolution_time']:.2f}")
        
    except Exception as e:
        logging.error(f"Erreur lors de l'exécution: {str(e)}")
        raise

if __name__ == "__main__":
    main()