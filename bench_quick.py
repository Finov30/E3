import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Food101
from bench_config import device, transform, models_to_test, QUICK_CONFIG
from bench_utils import train_model, evaluate_model
from visualize_results import save_and_visualize_results
from dataset_manager import DatasetManager
import pandas as pd

print(f"Using device: {device}")

# Initialisation et vérification du dataset
dataset_manager = DatasetManager(transform)
dataset_manager.check_and_prepare_dataset()

# Chargement des dataloaders
train_loader, test_loader = dataset_manager.get_dataloaders(
    batch_size=QUICK_CONFIG["batch_size"],
    num_samples=QUICK_CONFIG["num_samples"]
)

# Benchmark des modèles
results = {}
for model_name, model in models_to_test.items():
    print(f"\nTraining {model_name}...")
    try:
        print(f"Début de l'entraînement de {model_name}")
        training_time = train_model(model, train_loader, device, 
                                  epochs=QUICK_CONFIG["epochs"],
                                  model_name=model_name)
        print(f"Entraînement terminé. Temps: {training_time}")
        
        print(f"Début de l'évaluation de {model_name}")
        metrics = evaluate_model(model, test_loader, device, 
                               model_name=model_name)
        print(f"Métriques obtenues: {metrics}")
        
        # Création du dictionnaire de résultats avec les bonnes conversions
        results[model_name] = {
            "Training Time (s)": float(training_time),
            "Accuracy (%)": float(metrics.get("test_accuracy", 0)),
            "F1 Score (%)": float(metrics.get("f1_score", 0)),
            "Recall (%)": float(metrics.get("recall_score", 0))
        }
        print(f"Résultats pour {model_name}: {results[model_name]}")
        
        # Vérification que toutes les valeurs sont des nombres
        for key, value in results[model_name].items():
            if not isinstance(value, (int, float)) and key != "Model":
                print(f"Warning: {key} n'est pas un nombre: {value}")
                results[model_name][key] = 0.0

    except Exception as e:
        print(f"Erreur détaillée lors de l'évaluation de {model_name}:")
        import traceback
        traceback.print_exc()
        print(f"Message d'erreur: {str(e)}")
        results[model_name] = {
            "Training Time (s)": 0.0,
            "Accuracy (%)": 0.0,
            "F1 Score (%)": 0.0,
            "Recall (%)": 0.0
        }

print("\nBenchmark Results:")
for model, metrics in results.items():
    print(f"{model}: {metrics}")

try:
    save_and_visualize_results(results, benchmark_type="quick")
except Exception as e:
    print(f"Erreur lors de la visualisation: {e}")
    # Affichage des valeurs problématiques
    for model, metrics in results.items():
        print(f"\nModèle: {model}")
        for key, value in metrics.items():
            print(f"{key}: {value} (type: {type(value)})") 