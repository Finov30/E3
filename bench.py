import torch
from bench_config import device, transform, models_to_test, FULL_CONFIG
from bench_utils import train_model, evaluate_model, analyze_similar_classes, save_trained_model, save_and_visualize_results
from dataset_manager import DatasetManager
from model_monitor import ModelMonitor
import traceback
import numpy as np
import mlflow

print(f"Using device: {device}")

# Initialisation et vérification du dataset
dataset_manager = DatasetManager(transform)
if dataset_manager.dataset_exists():
    print("✓ Dataset déjà présent, chargement direct...")
else:
    print("Dataset non trouvé, téléchargement en cours...")
    dataset_manager.check_and_prepare_dataset()

# Chargement des dataloaders avec la configuration complète
train_loader, test_loader = dataset_manager.get_dataloaders(
    batch_size=FULL_CONFIG["batch_size"],
    num_samples=FULL_CONFIG["num_samples"]  # None pour utiliser tout le dataset
)

# Benchmark du modèle
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
        metrics = evaluate_model(model, test_loader, device, model_name=model_name)
        
        # Vérification des métriques
        if all(isinstance(v, (int, float)) and not np.isnan(v) for v in metrics.values()):
            results[model_name] = {
                "Training Time (s)": float(training_time),
                "Accuracy (%)": float(metrics["test_accuracy"]),
                "F1 Score (%)": float(metrics["f1_score"]),
                "Recall (%)": float(metrics["recall_score"]),
                "Log Loss": float(metrics["log_loss"]),
                "Top-3 Accuracy (%)": float(metrics["top_3_accuracy"]),
                "Top-5 Accuracy (%)": float(metrics["top_5_accuracy"])
            }
            print(f"✓ Résultats pour {model_name}: {results[model_name]}")
            
            # Sauvegarde du modèle entraîné
            try:
                saved_path = save_trained_model(model, model_name, results[model_name])
                if saved_path:
                    print(f"✓ Modèle sauvegardé avec succès dans: {saved_path}")
                    # Initialisation du monitor ici
                    monitor = ModelMonitor(model_name, run_type="evaluation")
                    if monitor.mlflow_active:
                        mlflow.log_artifact(saved_path, "saved_models")
                    monitor.close()
            except Exception as save_error:
                print(f"❌ Erreur lors de la sauvegarde du modèle {model_name}: {save_error}")
            
            # Sauvegarde immédiate des résultats partiels
            save_and_visualize_results({model_name: results[model_name]}, 
                                     benchmark_type="partial",
                                     suffix=f"_{model_name}")
        else:
            print(f"⚠️ Métriques invalides pour {model_name}")
        
        # Après l'évaluation du modèle
        print("\nAnalyse des classes similaires:")
        analyze_similar_classes(model, test_loader, device, model_name)
        
    except Exception as e:
        print(f"❌ Erreur détaillée lors de l'évaluation de {model_name}:")
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

# Sauvegarde finale des résultats
try:
    save_and_visualize_results(results, benchmark_type="full")
    print("\n✓ Tous les résultats ont été sauvegardés avec succès")
except Exception as e:
    print(f"\n❌ Erreur lors de la sauvegarde finale des résultats: {e}")



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