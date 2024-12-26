import pandas as pd
from utils import prepare_data, lazy_evaluate_models
import logging

# Configurer la journalisation
logging.basicConfig(filename='ticket_classification.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

try:
    # Charger le dataset
    df = pd.read_csv('C:\\Users\\samua\\Desktop\\Project E2 V4\\customer_support_tickets.csv')
    logging.info('Dataset chargé avec succès.')

    # Préparer les données
    X_train, X_test, y_train, y_test, label_encoder = prepare_data(df)
    logging.info('Données préparées avec succès.')

    # Entraîner et évaluer les modèles avec LazyPredict
    models, predictions, vectorizer = lazy_evaluate_models(X_train, X_test, y_train, y_test)
    logging.info('Évaluation des modèles terminée.')

    print("Résultats de l'évaluation des modèles avec LazyPredict:\n")
    print(models)  # Affiche la perf de chaque modèle

except Exception as e:
    logging.error(f'Une erreur est survenue : {e}')
    print(f'Une erreur est survenue : {e}')