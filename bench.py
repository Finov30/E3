import time
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import Food101
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
from visualize_results import save_and_visualize_results
from bench_config import device, transform, models_to_test, FULL_CONFIG
from bench_utils import train_model, evaluate_model
from dataset_manager import DatasetManager

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
    training_time = train_model(model, train_loader, device, epochs=FULL_CONFIG["epochs"])
    accuracy = evaluate_model(model, test_loader, device)
    results[model_name] = {"Training Time (s)": training_time, "Accuracy (%)": accuracy}

print("\nBenchmark Results:")
for model, metrics in results.items():
    print(f"{model}: {metrics}")

save_and_visualize_results(results, benchmark_type="full")
