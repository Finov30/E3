import torch
from bench_config import device, transform, models_to_test, FULL_CONFIG
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
    batch_size=FULL_CONFIG["batch_size"],
    num_samples=FULL_CONFIG["num_samples"]
)

# Benchmark des modèles
results = {}
for model_name, model in models_to_test.items():
    print(f"\nTraining {model_name}...")
    try:
        print(f"Début de l'entraînement de {model_name}")
        training_time = train_model(model, train_loader, device, 
                                  epochs=FULL_CONFIG["epochs"],
                                  model_name=model_name)
        print(f"Entraînement terminé. Temps: {training_time}")
        
        print(f"Début de l'évaluation de {model_name}")
        metrics = evaluate_model(model, test_loader, device, 
                               model_name=model_name)
        print(f"Métriques obtenues: {metrics}")
        
        # Création du dictionnaire de résultats avec les bonnes conversions
        results[model_name] = {
            "Training Time (s)": float(training_time),
            "Accuracy (%)": float(metrics["test_accuracy"]),
            "F1 Score (%)": float(metrics["f1_score"]),
            "Recall (%)": float(metrics["recall_score"]),
            "Log Loss": float(metrics["log_loss"]),
            "Top-3 Accuracy (%)": float(metrics["top_3_accuracy"]),
            "Top-5 Accuracy (%)": float(metrics["top_5_accuracy"])
        }
        print(f"Résultats pour {model_name}: {results[model_name]}")
        
    except Exception as e:
        print(f"Erreur détaillée lors de l'évaluation de {model_name}:")
        import traceback
        traceback.print_exc()
        print(f"Message d'erreur: {str(e)}")
        results[model_name] = {
            "Training Time (s)": 0.0,
            "Accuracy (%)": 0.0,
            "F1 Score (%)": 0.0,
            "Recall (%)": 0.0,
            "Log Loss": 0.0,
            "Top-3 Accuracy (%)": 0.0,
            "Top-5 Accuracy (%)": 0.0
        }

print("\nBenchmark Results:")
for model, metrics in results.items():
    print(f"{model}: {metrics}")

try:
    save_and_visualize_results(results, benchmark_type="full")
except Exception as e:
    print(f"Erreur lors de la visualisation: {e}")
    # Affichage des valeurs problématiques
    for model, metrics in results.items():
        print(f"\nModèle: {model}")
        for key, value in metrics.items():
            print(f"{key}: {value} (type: {type(value)})")
