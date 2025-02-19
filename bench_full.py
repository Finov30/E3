import torch
from bench_config import device, transform, models_to_test, FULL_CONFIG
from bench_utils import train_model, evaluate_model, analyze_similar_classes
from visualize_results import save_and_visualize_results
from dataset_manager import DatasetManager
import traceback

print(f"Using device: {device}")

# Initialisation et vérification du dataset
dataset_manager = DatasetManager(transform)
dataset_manager.check_and_prepare_dataset()

# Chargement des dataloaders avec la configuration complète
train_loader, test_loader = dataset_manager.get_dataloaders(
    batch_size=FULL_CONFIG["batch_size"],
    num_samples=FULL_CONFIG["num_samples"]  # None pour utiliser tout le dataset
)

def check_class_distribution(train_loader):
    class_counts = {}
    for _, labels in train_loader:
        for label in labels:
            label = label.item()
            class_counts[label] = class_counts.get(label, 0) + 1
    return class_counts

# Ajouter avant l'entraînement
class_distribution = check_class_distribution(train_loader)
print("Distribution des classes:", class_distribution)

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
        
        # Analyse des classes similaires
        print("\nAnalyse des classes similaires:")
        analyze_similar_classes(model, test_loader, device, model_name)
        
        # Stockage des résultats avec conversion explicite en float
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

        # Sauvegarde intermédiaire après chaque modèle
        try:
            save_and_visualize_results({k: results[k] for k in results.keys() if k <= model_name}, 
                                     benchmark_type="full",
                                     suffix=f"_intermediate_{model_name}")
        except Exception as viz_error:
            print(f"Erreur lors de la sauvegarde intermédiaire pour {model_name}: {viz_error}")

    except Exception as e:
        print(f"Erreur détaillée lors de l'évaluation de {model_name}:")
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
    # Sauvegarde et visualisation finale
    save_and_visualize_results(results, benchmark_type="full")
except Exception as e:
    print(f"Erreur lors de la visualisation finale: {e}")
    traceback.print_exc()
    # Affichage des valeurs problématiques
    for model, metrics in results.items():
        print(f"\nModèle: {model}")
        for key, value in metrics.items():
            print(f"{key}: {value} (type: {type(value)})")

    # Tentative de sauvegarde en CSV uniquement
    try:
        import pandas as pd
        from datetime import datetime
        
        df = pd.DataFrame.from_dict(results, orient='index')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f'benchmark_results/benchmark_full_{timestamp}.csv'
        df.to_csv(csv_path)
        print(f"\nLes résultats ont été sauvegardés en CSV: {csv_path}")
    except Exception as csv_error:
        print(f"Erreur lors de la sauvegarde en CSV: {csv_error}") 