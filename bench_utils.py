import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
from model_monitor import ModelMonitor
from mlflow_registry import ModelRegistry
import mlflow
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, recall_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

def train_model(model, train_loader, device, epochs=1, model_name=None):
    model.to(device)
    
    # Initialisation du monitoring
    monitor = ModelMonitor(model_name or model.__class__.__name__, run_type="training")
    monitor.log_model_architecture(model)
    
    # Modification de la dernière couche en fonction du type de modèle
    if hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, 101)
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, 101)
        else:
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, 101)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Log des hyperparamètres
    hyperparams = {
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "epochs": epochs,
        "batch_size": train_loader.batch_size,
        "device": str(device)
    }
    if monitor.mlflow_active:
        mlflow.log_params(hyperparams)
    
    start_time = time.time()
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Calcul de l'accuracy par batch
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            
            # Monitoring par batch
            metrics = {
                "batch_loss": loss.item(),
                "batch_accuracy": accuracy,
                "learning_rate": optimizer.param_groups[0]['lr']
            }
            monitor.log_batch_metrics(metrics, step=epoch * len(train_loader) + batch_idx)
            
            running_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'accuracy': f'{accuracy:.2f}%'
            })
        
        # Métriques de fin d'époque
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        epoch_metrics = {
            "epoch_loss": epoch_loss,
            "epoch_accuracy": epoch_accuracy,
            "epoch": epoch + 1
        }
        monitor.log_epoch_summary(epoch + 1, epoch_metrics)
        
        print(f"\nEpoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
    
    training_time = time.time() - start_time
    
    # Métriques finales
    final_metrics = {
        "total_training_time": training_time,
        "final_loss": epoch_loss,
        "final_accuracy": epoch_accuracy,
        "total_epochs": epochs
    }
    monitor.log_metrics(final_metrics)
    monitor.close()
    return training_time

def evaluate_model(model, test_loader, device, model_name=None):
    monitor = ModelMonitor(model_name or model.__class__.__name__, run_type="evaluation")
    
    try:
        model.to(device)
        model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Evaluation')
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Stockage pour les métriques détaillées
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())
                
                # Monitoring par batch
                accuracy = 100 * correct / total
                metrics = {
                    "batch_accuracy": accuracy,
                    "batch_size": labels.size(0)
                }
                monitor.log_batch_metrics(metrics, step=batch_idx)
                
                pbar.set_postfix({'accuracy': f'{accuracy:.2f}%'})
        
        # Calcul des métriques avancées
        final_accuracy = 100 * correct / total
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        
        # Classification report
        class_report = classification_report(all_labels, all_predictions)
        
        # Confusion Matrix
        try:
            conf_matrix = confusion_matrix(all_labels, all_predictions)
            
            # Visualisation de la matrice de confusion
            plt.figure(figsize=(15, 15))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            
            # Sauvegarde de la matrice de confusion
            os.makedirs('benchmark_results', exist_ok=True)
            confusion_matrix_path = f'benchmark_results/confusion_matrix_{model_name}.png'
            plt.savefig(confusion_matrix_path)
            plt.close()
        except Exception as e:
            print(f"Attention: Impossible de générer la matrice de confusion: {e}")
        
        # Métriques finales
        final_metrics = {
            "test_accuracy": float(final_accuracy),
            "f1_score": float(f1 * 100),
            "recall_score": float(recall * 100),
            "total_samples": int(total),
            "model_name": str(model_name or model.__class__.__name__)
        }
        
        # Log des métriques et rapports
        if monitor.mlflow_active:
            # S'assurer que toutes les valeurs sont des float
            numeric_metrics = {k: float(v) for k, v in final_metrics.items() 
                             if isinstance(v, (int, float))}
            mlflow.log_metrics(numeric_metrics)
            
        monitor.log_text("classification_report.txt", class_report)
        
        return final_metrics
        
    except Exception as e:
        print(f"Erreur dans evaluate_model: {e}")
        raise
    finally:
        # S'assurer que le monitor est toujours fermé
        monitor.close()
        if mlflow.active_run():
            mlflow.end_run()

def compare_models_performance():
    """Compare les performances des modèles enregistrés"""
    registry = ModelRegistry()
    models = ["ResNet-50", "EfficientNet-B0", "MobileNetV3"]
    
    print("\nComparaison des modèles enregistrés:")
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