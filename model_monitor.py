import logging
import os
from datetime import datetime
import mlflow
from tensorboard.backend.event_processing import event_accumulator
import torch
from torch.utils.tensorboard import SummaryWriter
import psutil
import platform
import json
import subprocess
import time
import socket
from mlflow_registry import ModelRegistry

class ModelMonitor:
    def __init__(self, model_name, run_type="training"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_name = model_name
        self.log_dir = os.path.join("logs", model_name, self.timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Configuration du logger
        self.logger = logging.getLogger(f"{model_name}_{run_type}")
        self.logger.setLevel(logging.INFO)
        
        # Handlers
        fh = logging.FileHandler(os.path.join(self.log_dir, f"{run_type}.log"))
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        for handler in [fh, ch]:
            handler.setLevel(logging.INFO)
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # TensorBoard
        try:
            self.writer = SummaryWriter(log_dir=os.path.join("runs", model_name, self.timestamp))
            self.logger.info("TensorBoard initialisé avec succès")
        except Exception as e:
            self.logger.warning(f"Erreur lors de l'initialisation de TensorBoard: {e}")
            self.writer = None
        
        # MLflow
        try:
            if mlflow.get_tracking_uri():
                # Création/récupération de l'experiment
                experiment = mlflow.get_experiment_by_name(model_name)
                if experiment is None:
                    mlflow.create_experiment(model_name)
                mlflow.set_experiment(model_name)
                
                # Démarrage du run avec tags
                self.run = mlflow.start_run(
                    run_name=f"{run_type}_{self.timestamp}",
                    tags={
                        "model_name": model_name,
                        "run_type": run_type,
                        "timestamp": self.timestamp
                    }
                )
                
                # Log des informations de base
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("run_type", run_type)
                
                self.logger.info("MLflow initialisé avec succès")
                self.mlflow_active = True
            else:
                self.logger.warning("MLflow tracking URI non définie")
                self.mlflow_active = False
        except Exception as e:
            self.logger.warning(f"Erreur lors de l'initialisation de MLflow: {e}")
            self.mlflow_active = False
        
        self._log_environment()
        self.model_registry = ModelRegistry()

    def log_metrics(self, metrics, step=None):
        """Log des métriques générales"""
        try:
            # Log dans TensorBoard
            if self.writer:
                for name, value in metrics.items():
                    if isinstance(value, (int, float)):  # Vérifie que la valeur est numérique
                        self.writer.add_scalar(f'metrics/{name}', value, step)

            # Log dans MLflow
            if self.mlflow_active:
                # Filtrer et convertir les métriques pour MLflow
                numeric_metrics = {}
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        numeric_metrics[k] = float(v)  # Conversion explicite en float
                    elif isinstance(v, str):
                        # Les strings doivent être loggés comme des paramètres, pas des métriques
                        mlflow.log_param(k, v)
                
                if numeric_metrics:
                    if step is not None:
                        mlflow.log_metrics(numeric_metrics, step=step)
                    else:
                        mlflow.log_metrics(numeric_metrics)

            # Log dans le logger
            self.logger.info(f"Metrics at step {step}: {metrics}")

        except Exception as e:
            self.logger.error(f"Erreur lors du logging des métriques: {e}")

    def close(self):
        """Ferme proprement toutes les connexions"""
        try:
            if self.writer:
                self.writer.close()
                self.logger.info("TensorBoard writer fermé")

            if self.mlflow_active:
                # Log des artifacts avant de fermer
                if os.path.exists(self.log_dir):
                    mlflow.log_artifacts(self.log_dir, artifact_path="logs")
                    self.logger.info("Artifacts MLflow enregistrés")
                mlflow.end_run()
                self.logger.info("MLflow run terminé")

        except Exception as e:
            self.logger.error(f"Erreur lors de la fermeture des connexions: {e}")

    def log_model(self, model, model_name):
        """Log le modèle avec MLflow et l'enregistre dans le registre"""
        if self.mlflow_active:
            try:
                # Log du modèle avec MLflow
                mlflow.pytorch.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=model_name
                )
                self.logger.info(f"Modèle {model_name} enregistré dans MLflow")

                # Enregistrement dans le registre
                version = self.model_registry.register_model(
                    model_name=model_name,
                    run_id=self.run.info.run_id,
                    description=f"Model trained on {self.timestamp}"
                )

                if version:
                    # Si c'est un run d'évaluation, on peut automatiquement promouvoir le modèle
                    if hasattr(self, 'final_metrics') and 'test_accuracy' in self.final_metrics:
                        accuracy = self.final_metrics['test_accuracy']
                        if accuracy > 0.75:  # Seuil exemple
                            self.model_registry.transition_model_version_stage(
                                model_name, version, "Production"
                            )
                        elif accuracy > 0.5:
                            self.model_registry.transition_model_version_stage(
                                model_name, version, "Staging"
                            )

                # Log des paramètres du modèle
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                model_params = {
                    "model_name": model_name,
                    "model_version": version,
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "model_type": model.__class__.__name__
                }
                mlflow.log_params(model_params)

            except Exception as e:
                self.logger.error(f"Erreur lors du logging du modèle: {e}")

    def log_model_architecture(self, model):
        """Log l'architecture du modèle dans MLflow et TensorBoard"""
        try:
            # Création d'une description textuelle du modèle
            model_summary = str(model)
            
            # Log dans MLflow
            if self.mlflow_active:
                # Log de l'architecture comme texte
                mlflow.log_text(model_summary, "model_architecture.txt")
                
                # Log des paramètres du modèle avec le nom du modèle
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                architecture_params = {
                    "model_name": self.model_name,  # Utilisation du nom stocké
                    "model_type": model.__class__.__name__,
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "layers_count": len(list(model.modules()))
                }
                mlflow.log_params(architecture_params)
                
                # Ajout d'un tag pour faciliter la recherche
                mlflow.set_tag("model_architecture", model.__class__.__name__)
                
                # Log du graphe du modèle si possible
                try:
                    dummy_input = torch.randn(1, 3, 224, 224)
                    if torch.cuda.is_available():
                        dummy_input = dummy_input.cuda()
                        model = model.cuda()
                    self.writer.add_graph(model, dummy_input)
                    self.logger.info("Graphe du modèle ajouté à TensorBoard")
                except Exception as e:
                    self.logger.warning(f"Impossible d'ajouter le graphe du modèle: {e}")
            
            # Log dans le fichier local
            architecture_file = os.path.join(self.log_dir, "model_architecture.txt")
            with open(architecture_file, "w") as f:
                f.write(f"Model: {self.model_name}\n")
                f.write("=" * 50 + "\n")
                f.write(model_summary)
                f.write("\n\nParamètres du modèle:\n")
                f.write(f"Total parameters: {total_params}\n")
                f.write(f"Trainable parameters: {trainable_params}\n")
            
            self.logger.info(f"Architecture du modèle {self.model_name} enregistrée dans {architecture_file}")
            
        except Exception as e:
            self.logger.error(f"Erreur lors du logging de l'architecture: {e}")

    def _log_environment(self):
        """Log des informations sur l'environnement d'exécution"""
        try:
            env_info = {
                "python_version": platform.python_version(),
                "system": platform.system(),
                "processor": platform.processor(),
                "memory_total": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
                "memory_available": f"{psutil.virtual_memory().available / (1024**3):.2f} GB",
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
                "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "timestamp": self.timestamp
            }
            
            # Log dans MLflow
            if self.mlflow_active:
                mlflow.log_params(env_info)
            
            # Log dans le fichier local
            env_file = os.path.join(self.log_dir, "environment_info.json")
            with open(env_file, "w") as f:
                json.dump(env_info, f, indent=4)
            
            # Log dans le logger
            self.logger.info("Environment info:")
            for key, value in env_info.items():
                self.logger.info(f"  {key}: {value}")
            
        except Exception as e:
            self.logger.error(f"Erreur lors du logging de l'environnement: {e}")

    def log_batch_metrics(self, metrics, step):
        """Log des métriques détaillées par batch"""
        try:
            # Log dans TensorBoard
            if self.writer:
                for name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f'batch/{name}', value, step)
            
            # Log dans MLflow
            if self.mlflow_active:
                batch_metrics = {
                    f"batch_{k}": float(v) if isinstance(v, (int, float)) else v
                    for k, v in metrics.items()
                    if not isinstance(v, str)
                }
                if batch_metrics:
                    mlflow.log_metrics(batch_metrics, step=step)
            
            # Log dans le logger avec un niveau de détail réduit pour éviter trop de logs
            if step % 10 == 0:  # Log tous les 10 batches
                self.logger.debug(f"Batch metrics at step {step}: {metrics}")
            
        except Exception as e:
            self.logger.error(f"Erreur lors du logging des métriques de batch: {e}")

    def log_epoch_summary(self, epoch, metrics):
        """Log un résumé de l'époque"""
        try:
            # Log dans TensorBoard
            if self.writer:
                for name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f'epoch/{name}', value, epoch)
            
            # Log dans MLflow
            if self.mlflow_active:
                epoch_metrics = {
                    f"epoch_{k}": float(v) if isinstance(v, (int, float)) else v
                    for k, v in metrics.items()
                    if not isinstance(v, str)
                }
                if epoch_metrics:
                    mlflow.log_metrics(epoch_metrics, step=epoch)
            
            # Log dans le logger
            self.logger.info(f"Epoch {epoch} summary: {metrics}")
            
        except Exception as e:
            self.logger.error(f"Erreur lors du logging du résumé de l'époque: {e}")

    def log_text(self, filename, content):
        """Log du texte dans un fichier"""
        try:
            # Log dans MLflow
            if self.mlflow_active:
                mlflow.log_text(content, filename)
            
            # Log dans le fichier local
            file_path = os.path.join(self.log_dir, filename)
            with open(file_path, "w") as f:
                f.write(content)
            
            self.logger.info(f"Texte enregistré dans {filename}")
            
        except Exception as e:
            self.logger.error(f"Erreur lors du logging du texte: {e}")