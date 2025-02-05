import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Food101
from bench_config import device, transform, models_to_test, QUICK_CONFIG
from bench_utils import train_model, evaluate_model
from visualize_results import save_and_visualize_results

print(f"Using device: {device}")

# Chargement du dataset Food-101
dataset_train = Food101(root="./data", split="train", transform=transform, download=True)
dataset_test = Food101(root="./data", split="test", transform=transform, download=True)

# Réduction du dataset pour les tests rapides
dataset_train = torch.utils.data.Subset(dataset_train, range(QUICK_CONFIG["num_samples"]))
dataset_test = torch.utils.data.Subset(dataset_test, range(QUICK_CONFIG["num_samples"]//2))

train_loader = DataLoader(dataset_train, batch_size=QUICK_CONFIG["batch_size"], shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=QUICK_CONFIG["batch_size"], shuffle=False)

# Benchmark des modèles
results = {}
for model_name, model in models_to_test.items():
    print(f"\nTraining {model_name}...")
    training_time = train_model(model, train_loader, device, epochs=QUICK_CONFIG["epochs"])
    accuracy = evaluate_model(model, test_loader, device)
    results[model_name] = {"Training Time (s)": training_time, "Accuracy (%)": accuracy}

print("\nBenchmark Results:")
for model, metrics in results.items():
    print(f"{model}: {metrics}")

save_and_visualize_results(results, benchmark_type="quick") 