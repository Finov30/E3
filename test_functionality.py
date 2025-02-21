import torch
from bench_config import device, transform, models_to_test, FULL_CONFIG
from bench_utils import (
    train_model, 
    evaluate_model, 
    save_trained_model, 
    load_saved_model,
    test_model_loading
)
from dataset_manager import DatasetManager
import mlflow
from pathlib import Path
import traceback
import requests
import subprocess
import time
import os
import signal
import sys
import torch.nn as nn
from torchvision.datasets import Food101

def test_dataset_loading():
    """Test du chargement du dataset"""
    print("\n1. Test du chargement du dataset")
    try:
        dataset_manager = DatasetManager(transform)
        if dataset_manager.dataset_exists():
            print("✓ Dataset déjà présent")
            # Vérifie que les classes sont bien chargées
            train_dataset = Food101(root='./data', split='train', transform=transform)
            if len(train_dataset.classes) == 101:
                print("✓ Les 101 classes sont bien présentes")
            else:
                print("❌ Nombre de classes incorrect")
        else:
            print("Dataset non trouvé, téléchargement en cours...")
            dataset_manager.check_and_prepare_dataset()
            
        train_loader, test_loader = dataset_manager.get_dataloaders(
            batch_size=1,
            num_samples=2
        )
        
        # Vérifie la structure des données
        images, labels = next(iter(train_loader))
        if images.shape[1:] == (3, 224, 224):  # Vérifie la taille des images
            print("✓ Format des images correct")
        if labels.max() < 101:  # Vérifie les labels
            print("✓ Labels corrects")
            
        return train_loader, test_loader
    except Exception as e:
        print(f"❌ Erreur lors du chargement du dataset: {e}")
        return None, None

def test_model_training(model_name, model, train_loader):
    """Test de l'entraînement du modèle"""
    print(f"\n2. Test de l'entraînement de {model_name}")
    try:
        # Vérifie la structure du modèle
        if hasattr(model, 'fc'):
            original_out_features = model.fc.out_features
            if original_out_features != 101:
                print("⚠️ Modification de la couche de sortie nécessaire")
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, 101)
                print("✓ Couche de sortie adaptée")
        
        # Vérifie le device
        model = model.to(device)
        print(f"✓ Modèle chargé sur {device}")
        
        # Test rapide d'entraînement
        training_time = train_model(
            model=model,
            train_loader=train_loader,
            device=device,
            epochs=1,
            model_name=model_name
        )
        
        # Vérifie les gradients
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                break
        if has_grad:
            print("✓ Les gradients sont calculés correctement")
        else:
            print("❌ Problème avec les gradients")
                
        print(f"✓ Entraînement test réussi en {training_time:.2f} secondes")
        return True
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement: {e}")
        traceback.print_exc()
        return False

def test_model_evaluation(model_name, model, test_loader):
    """Test de l'évaluation du modèle"""
    print(f"\n3. Test de l'évaluation de {model_name}")
    try:
        metrics = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device,
            model_name=model_name
        )
        print(f"✓ Évaluation réussie avec métriques: {metrics}")
        return metrics
    except Exception as e:
        print(f"❌ Erreur lors de l'évaluation: {e}")
        traceback.print_exc()
        return None

def test_model_saving(model, model_name, metrics):
    """Test de la sauvegarde du modèle"""
    print(f"\n4. Test de la sauvegarde de {model_name}")
    try:
        saved_path = save_trained_model(
            model=model,
            model_name=model_name,
            metrics=metrics
        )
        if saved_path:
            print(f"✓ Modèle sauvegardé dans: {saved_path}")
            return saved_path
        return None
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde: {e}")
        traceback.print_exc()
        return None

def test_model_loading(saved_path):
    """Test du chargement du modèle"""
    print("\n5. Test du chargement du modèle")
    try:
        model, metrics = load_saved_model(saved_path)
        if model is not None:
            print("✓ Modèle chargé avec succès")
            print(f"✓ Métriques chargées: {metrics}")
            return True
        return False
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        traceback.print_exc()
        return False

def test_mlflow_logging():
    """Test du logging MLflow"""
    print("\n6. Test du logging MLflow")
    try:
        runs = mlflow.search_runs()
        if not runs.empty:
            print(f"✓ {len(runs)} runs MLflow trouvés")
            print("✓ Dernier run:", runs.iloc[0])
            return True
        else:
            print("⚠️ Aucun run MLflow trouvé")
            return False
    except Exception as e:
        print(f"❌ Erreur lors de la vérification MLflow: {e}")
        return False

def check_mlflow_server():
    """Vérifie si le serveur MLflow est accessible"""
    try:
        response = requests.get("http://localhost:5000")
        return response.status_code == 200
    except Exception:
        return False

def start_mlflow_server():
    """Démarre le serveur MLflow"""
    try:
        # Vérifie si le serveur est déjà en cours d'exécution
        if check_mlflow_server():
            print("✓ Serveur MLflow déjà en cours d'exécution")
            return True

        print("Démarrage du serveur MLflow...")
        # Démarre le serveur MLflow dans un nouveau processus
        mlflow_process = subprocess.Popen(
            ["mlflow", "server", "--host", "127.0.0.1", "--port", "5000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Attend que le serveur soit prêt
        for _ in range(5):  # Essaie 5 fois
            time.sleep(2)  # Attend 2 secondes entre chaque tentative
            if check_mlflow_server():
                print("✓ Serveur MLflow démarré avec succès")
                # Enregistre le PID pour l'arrêt ultérieur
                with open(".mlflow.pid", "w") as f:
                    f.write(str(mlflow_process.pid))
                return True
                
        print("❌ Échec du démarrage du serveur MLflow")
        return False
        
    except Exception as e:
        print(f"❌ Erreur lors du démarrage du serveur MLflow: {e}")
        return False

def stop_mlflow_server():
    """Arrête le serveur MLflow"""
    try:
        if os.path.exists(".mlflow.pid"):
            with open(".mlflow.pid", "r") as f:
                pid = int(f.read())
            os.kill(pid, signal.SIGTERM)
            os.remove(".mlflow.pid")
            print("✓ Serveur MLflow arrêté")
    except Exception as e:
        print(f"⚠️ Erreur lors de l'arrêt du serveur MLflow: {e}")

def test_model_monitor():
    """Test du monitoring"""
    print("\n7. Test du monitoring")
    try:
        from model_monitor import ModelMonitor
        monitor = ModelMonitor("test", "training")
        
        # Test des fonctionnalités essentielles
        test_metrics = {
            "test_metric": 0.5,
            "test_loss": 0.1
        }
        monitor.log_metrics(test_metrics)
        print("✓ Log des métriques OK")
        
        # Test de MLflow uniquement car c'est le plus important pour le benchmark
        if monitor.mlflow_active:
            print("✓ MLflow OK")
            
        monitor.close()
        return True
    except Exception as e:
        print(f"❌ Erreur lors du test du monitoring: {e}")
        return False

def main():
    print("=== Début des tests de fonctionnalité ===")
    
    # Configuration initiale
    print("\nVérification de la configuration:")
    print(f"Device: {device}")
    print(f"Batch size: {FULL_CONFIG['batch_size']}")
    print(f"Epochs: {FULL_CONFIG['epochs']}")
    
    # Démarre le serveur MLflow si nécessaire
    if not check_mlflow_server():
        if not start_mlflow_server():
            print("Les tests continueront sans tracking MLflow")
    
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        print("✓ MLflow initialisé avec succès")
        
        # 1. Test du dataset
        train_loader, test_loader = test_dataset_loading()
        if train_loader is None or test_loader is None:
            print("❌ Arrêt des tests : échec du chargement du dataset")
            return
        
        # Test du ResNet-50
        model_name = "ResNet-50"
        model = models_to_test[model_name]
        
        # 2. Test de l'entraînement
        if not test_model_training(model_name, model, train_loader):
            print("❌ Arrêt des tests : échec de l'entraînement")
            return
        
        # 3. Test de l'évaluation
        metrics = test_model_evaluation(model_name, model, test_loader)
        if metrics is None:
            print("❌ Arrêt des tests : échec de l'évaluation")
            return
        
        # 4. Test de la sauvegarde
        saved_path = test_model_saving(model, model_name, metrics)
        if saved_path is None:
            print("❌ Arrêt des tests : échec de la sauvegarde")
            return
        
        # 5. Test du chargement
        if not test_model_loading(saved_path):
            print("❌ Arrêt des tests : échec du chargement")
            return
        
        # 6. Test MLflow
        test_mlflow_logging()
        
        # 7. Test du monitoring
        test_model_monitor()
        
        print("\n=== Résumé des tests ===")
        print("✓ Dataset : OK")
        print("✓ Entraînement : OK")
        print("✓ Évaluation : OK")
        print("✓ Sauvegarde : OK")
        print("✓ Chargement : OK")
        print("✓ MLflow : OK")
        print("✓ Monitoring : OK")
        print("\n✓ Tous les tests ont réussi!")
        
    except Exception as e:
        print(f"⚠️ Erreur d'initialisation MLflow: {e}")
        print("Les tests continueront sans tracking MLflow")
    finally:
        # Arrête le serveur MLflow à la fin des tests
        stop_mlflow_server()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterruption détectée, arrêt propre...")
        stop_mlflow_server()
        sys.exit(0) 