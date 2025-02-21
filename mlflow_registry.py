import mlflow
from mlflow.tracking import MlflowClient
import logging
from datetime import datetime

class ModelRegistry:
    def __init__(self):
        self.client = MlflowClient()
        self.logger = logging.getLogger("ModelRegistry")

    def register_model(self, model_name, run_id, description=None):
        """Enregistre un modèle dans le registre MLflow"""
        try:
            # Vérification si le modèle existe déjà
            try:
                model_details = self.client.get_registered_model(model_name)
                self.logger.info(f"Modèle {model_name} déjà existant dans le registre")
            except:
                # Création du modèle enregistré s'il n'existe pas
                model_details = self.client.create_registered_model(
                    name=model_name,
                    description=description or f"Modèle {model_name} créé le {datetime.now()}"
                )
                self.logger.info(f"Nouveau modèle {model_name} créé dans le registre")

            # Création d'une nouvelle version
            version = self.client.create_model_version(
                name=model_name,
                source=f"runs:/{run_id}/model",
                run_id=run_id
            )
            
            self.logger.info(f"Nouvelle version {version.version} créée pour {model_name}")
            return version.version

        except Exception as e:
            self.logger.error(f"Erreur lors de l'enregistrement du modèle: {e}")
            return None

    def transition_model_version_stage(self, model_name, version, stage):
        """Change le stage d'une version de modèle (None, Staging, Production, Archived)"""
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            self.logger.info(f"Modèle {model_name} version {version} transitionné vers {stage}")
            return True
        except Exception as e:
            self.logger.error(f"Erreur lors de la transition du modèle: {e}")
            return False

    def get_latest_versions(self, model_name, stages=None):
        """Récupère les dernières versions d'un modèle"""
        try:
            return self.client.get_latest_versions(model_name, stages)
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des versions: {e}")
            return []

    def get_best_model(self, model_name, metric="test_accuracy", stage="Production"):
        """Récupère la meilleure version d'un modèle basée sur une métrique"""
        try:
            versions = self.get_latest_versions(model_name, [stage])
            if not versions:
                return None

            best_version = None
            best_metric = float('-inf')

            for version in versions:
                run = mlflow.get_run(version.run_id)
                current_metric = float(run.data.metrics.get(metric, float('-inf')))
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_version = version

            return best_version
        except Exception as e:
            self.logger.error(f"Erreur lors de la recherche du meilleur modèle: {e}")
            return None

def compare_models_performance():
    """Compare les performances du modèle enregistré"""
    registry = ModelRegistry()
    models = ["ResNet-50"]
    
    print("\nComparaison du modèle enregistré:")
    print("=" * 50)
    
    for model_name in models:
        # Récupération des versions en production et staging
        prod_version = registry.get_best_model(model_name, stage="Production")
        staging_version = registry.get_best_model(model_name, stage="Staging")
        
        print(f"\nModèle: {model_name}")
        if prod_version:
            run = mlflow.get_run(prod_version.run_id)
            print(f"Production (v{prod_version.version}):")
            print(f"  - Accuracy: {run.data.metrics.get('test_accuracy', 'N/A'):.2f}%")
            print(f"  - Training Time: {run.data.metrics.get('total_training_time', 'N/A'):.2f}s")
        
        if staging_version:
            run = mlflow.get_run(staging_version.run_id)
            print(f"Staging (v{staging_version.version}):")
            print(f"  - Accuracy: {run.data.metrics.get('test_accuracy', 'N/A'):.2f}%")
            print(f"  - Training Time: {run.data.metrics.get('total_training_time', 'N/A'):.2f}s") 