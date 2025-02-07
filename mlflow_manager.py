import os
import mlflow
import subprocess
import socket
import time
import logging
from pathlib import Path

class MLflowManager:
    def __init__(self, tracking_uri="http://localhost:5000"):
        self.tracking_uri = tracking_uri
        self.logger = logging.getLogger("MLflowManager")
        
    @staticmethod
    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    def start_server(self):
        """Démarre le serveur MLflow s'il n'est pas déjà en cours d'exécution"""
        if not self.is_port_in_use(5000):
            try:
                print("Démarrage du serveur MLflow...")
                # Création du dossier mlruns s'il n'existe pas
                Path("mlruns").mkdir(exist_ok=True)
                
                # Démarrage du serveur MLflow
                subprocess.Popen(
                    ["mlflow", "ui", "--port", "5000"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                # Attente que le serveur soit prêt
                for _ in range(5):  # 5 tentatives
                    if self.is_port_in_use(5000):
                        print("✓ Serveur MLflow démarré avec succès")
                        return True
                    time.sleep(1)
                    
                print("✗ Échec du démarrage du serveur MLflow")
                return False
            except Exception as e:
                print(f"✗ Erreur lors du démarrage de MLflow: {e}")
                return False
        else:
            print("✓ Serveur MLflow déjà en cours d'exécution")
            return True

    def initialize(self):
        """Initialise MLflow pour le projet"""
        if self.start_server():
            try:
                # Configuration de l'URI de tracking
                mlflow.set_tracking_uri(self.tracking_uri)
                print(f"MLflow configuré avec URI: {self.tracking_uri}")
                
                # Vérification de la connexion
                mlflow.search_experiments()
                return True
            except Exception as e:
                print(f"Erreur lors de l'initialisation de MLflow: {e}")
                return False
        return False

    def create_experiment(self, model_name):
        """Crée ou récupère un experiment pour un modèle"""
        try:
            experiment = mlflow.get_experiment_by_name(model_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(model_name)
                print(f"Nouvel experiment créé pour {model_name}")
            else:
                experiment_id = experiment.experiment_id
                print(f"Experiment existant trouvé pour {model_name}")
            return experiment_id
        except Exception as e:
            print(f"Erreur lors de la création de l'experiment: {e}")
            return None 