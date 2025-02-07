import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Food101
from bench_config import device, transform, models_to_test, QUICK_CONFIG
from bench_utils import train_model, evaluate_model
from visualize_results import save_and_visualize_results
from dataset_manager import DatasetManager

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
    training_time = train_model(model, train_loader, device, 
                              epochs=QUICK_CONFIG["epochs"],
                              model_name=model_name)
    accuracy = evaluate_model(model, test_loader, device, 
                            model_name=model_name)
    results[model_name] = {"Training Time (s)": training_time, "Accuracy (%)": accuracy}

print("\nBenchmark Results:")
for model, metrics in results.items():
    print(f"{model}: {metrics}")

save_and_visualize_results(results, benchmark_type="quick") 