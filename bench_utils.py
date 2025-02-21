import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
from model_monitor import ModelMonitor
from mlflow_registry import ModelRegistry
import mlflow
import numpy as np
from sklearn.metrics import f1_score, recall_score, classification_report, confusion_matrix, precision_score
import seaborn as sns
import os
from torchvision import transforms
from bench_config import OPTIMIZER_CONFIG, models_to_test
from torchvision.datasets import Food101
from datetime import datetime
import pandas as pd
import traceback

import warnings
from pathlib import Path
import json

# Configuration des warnings Python standards
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

def train_model(model, train_loader, device, epochs=1, model_name=None):
    # S'assurer qu'aucune session MLflow n'est active
    if mlflow.active_run():
        mlflow.end_run()
        
    model.to(device)
    
    try:
        # Initialisation du monitoring
        monitor = ModelMonitor(model_name or model.__class__.__name__, run_type="training")
        monitor.log_model_architecture(model)
        
        # Ajout : Log de l'architecture du modèle
        if monitor.mlflow_active:
            mlflow.log_param("model_architecture", str(model))
            mlflow.log_param("total_parameters", sum(p.numel() for p in model.parameters()))
        
        # Modification de la dernière couche en fonction du type de modèle
        if hasattr(model, 'fc'):
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, 101)
            )
        elif hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Sequential):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(in_features, 101)
                )
            else:
                in_features = model.classifier.in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(in_features, 101)
                )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), 
                             lr=OPTIMIZER_CONFIG["learning_rate"],
                             momentum=OPTIMIZER_CONFIG["momentum"],
                             weight_decay=2e-4,
                             nesterov=True)
        
        # Modification du scheduler pour une stratégie plus agressive
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                        mode='min',
                                                        factor=0.1,
                                                        patience=2,
                                                        verbose=True)

        # Augmentation de données plus agressive
        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5)
        ])
        
        # Log des hyperparamètres
        hyperparams = {
            "learning_rate": OPTIMIZER_CONFIG["learning_rate"],
            "optimizer": "SGD",
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
            
            # Ajout du scheduler step
            scheduler.step(running_loss)
            
            # Log des métriques additionnelles
            monitor.log_metrics({
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch_loss": epoch_loss,
                "epoch_accuracy": epoch_accuracy
            })
            
            # Ajout du gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Monitoring des gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    monitor.log_metrics({
                        f"grad_{name}_norm": param.grad.norm().item()
                    })
        
        training_time = time.time() - start_time
        
        # Métriques finales
        final_metrics = {
            "total_training_time": training_time,
            "final_loss": epoch_loss,
            "final_accuracy": epoch_accuracy,
            "total_epochs": epochs
        }
        monitor.log_metrics(final_metrics)
        
        # Ajout : Log des courbes d'apprentissage
        if monitor.mlflow_active:
            mlflow.log_metrics({
                "final_training_loss": epoch_loss,
                "final_training_accuracy": epoch_accuracy,
                "total_training_time": training_time
            })
        
        return training_time
        
    except Exception as e:
        if monitor.mlflow_active:
            mlflow.log_param("training_error", str(e))
        raise
    finally:
        # S'assurer que la session est fermée
        monitor.close()
        if mlflow.active_run():
            mlflow.end_run()



def evaluate_model(model, test_loader, device, model_name=None):
    monitor = ModelMonitor(model_name or model.__class__.__name__, run_type="evaluation")
    
    try:
        model.to(device)
        model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        running_log_loss = 0.0
        top_k_correct = {1: 0, 3: 0, 5: 0}
        
        criterion = nn.CrossEntropyLoss(reduction='sum')
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Evaluation')
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                batch_log_loss = criterion(outputs, labels).item()
                running_log_loss += batch_log_loss
                
                for k in top_k_correct.keys():
                    _, top_k_pred = outputs.topk(k, 1, True, True)
                    top_k_correct[k] += torch.eq(top_k_pred, labels.view(-1, 1).expand_as(top_k_pred)).sum().item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                accuracy = 100 * correct / total
                
                # Log des métriques par batch dans MLflow
                if monitor.mlflow_active and batch_idx % 10 == 0:  # Log tous les 10 batches
                    monitor.log_batch_metrics({
                        "batch_accuracy": accuracy,
                        "batch_loss": batch_log_loss / labels.size(0)
                    }, batch_idx)
                
                pbar.set_postfix({'accuracy': f'{accuracy:.2f}%'})
        
        # Calcul des métriques finales
        final_accuracy = 100 * correct / total
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        log_loss = running_log_loss / total
        top_k_accuracy = {k: (100 * v / total) for k, v in top_k_correct.items()}
        
        # Ajout de vérifications pour les métriques
        final_metrics = {
            "test_accuracy": float(final_accuracy),
            "f1_score": float(f1 * 100),
            "recall_score": float(recall * 100),
            "precision_score": float(precision * 100),
            "log_loss": float(log_loss),
            "top_3_accuracy": float(top_k_accuracy[3]),
            "top_5_accuracy": float(top_k_accuracy[5]),
            "total_samples": total
        }
        
        # Vérification que toutes les métriques sont valides
        for metric_name, value in final_metrics.items():
            if not isinstance(value, (int, float)) or np.isnan(value):
                print(f"⚠️ Attention: Métrique {metric_name} invalide ({value})")
                final_metrics[metric_name] = 0.0
        
        if monitor.mlflow_active:
            mlflow.log_metrics(final_metrics)
            
            # Log des méta-informations supplémentaires
            mlflow.log_params({
                "model_name": model_name,
                "evaluation_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "device": str(device),
                "batch_size": test_loader.batch_size,
                "dataset_size": total
            })
            
            # Sauvegarde du rapport de classification détaillé
            classification_report_dict = classification_report(all_labels, all_predictions, output_dict=True)
            mlflow.log_dict(classification_report_dict, "classification_report.json")
            
            # Ajout : Log de la matrice de confusion
            conf_matrix = confusion_matrix(all_labels, all_predictions)
            mlflow.log_dict({"confusion_matrix": conf_matrix.tolist()}, 
                          "confusion_matrix.json")
            
            # Log des métriques par classe
            class_report = classification_report(all_labels, all_predictions, 
                                              output_dict=True)
            mlflow.log_dict(class_report, "class_metrics.json")
            
            # Log du timestamp de fin d'évaluation
            mlflow.log_param("evaluation_end_time", 
                           datetime.now().strftime("%Y%m%d_%H%M%S"))
        
        return final_metrics
        
    except Exception as e:
        if monitor.mlflow_active:
            mlflow.log_param("evaluation_error", str(e))
        raise
    finally:
        monitor.close()
        if mlflow.active_run():
            mlflow.end_run()

def analyze_similar_classes(model, test_loader, device, model_name, n_similar=5):
    """Analyse les classes souvent confondues entre elles"""
    model.eval()
    all_predictions = []
    all_labels = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Récupérer les classes depuis le dataset original
        # Modification ici : on met download=False car le dataset devrait déjà être téléchargé
        dataset = Food101(root='./data', split='test', download=False)
        class_names = dataset.classes
    except Exception as e:
        print(f"⚠️ Erreur lors de l'accès au dataset: {e}")
        try:
            # Si le dataset n'est pas trouvé, on essaie de le télécharger
            print("Tentative de téléchargement du dataset...")
            dataset = Food101(root='./data', split='test', download=True)
            class_names = dataset.classes
        except Exception as download_error:
            print(f"❌ Impossible d'accéder ou de télécharger le dataset: {download_error}")
            print("⚠️ Analyse des classes similaires impossible")
            return
    
    try:
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, top5_pred = outputs.topk(5, 1, True, True)
                all_predictions.extend(top5_pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Analyse des confusions fréquentes
        confusion_pairs = {}
        for true_label, top5 in zip(all_labels, all_predictions):
            true_class = class_names[true_label]
            for pred in top5:
                pred_class = class_names[pred]
                if true_class != pred_class:
                    pair = tuple(sorted([true_class, pred_class]))
                    confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
        
        # Analyse et sauvegarde des résultats
        most_confused = analyze_and_save_confusions(model_name, confusion_pairs, timestamp)
        
        if most_confused:
            print(f"\nClasse la plus problématique: {most_confused[0]} ({most_confused[1]} confusions)")
        
        # Affichage des paires les plus confondues
        print("\nPaires de classes fréquemment confondues:")
        for (class1, class2), count in sorted(confusion_pairs.items(), 
                                            key=lambda x: x[1], 
                                            reverse=True)[:n_similar]:
            print(f"{class1} <-> {class2}: {count} fois")
            
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse des classes similaires: {e}")
        traceback.print_exc()

def analyze_and_save_confusions(model_name, confusion_pairs, timestamp):
    """Analyse et sauvegarde les confusions fréquentes"""
    # Création des dossiers pour les analyses
    os.makedirs('confusion_analysis', exist_ok=True)
    os.makedirs('confusion_analysis/csv', exist_ok=True)
    
    # Compter les occurrences de chaque classe dans les confusions
    class_confusion_count = {}
    for (class1, class2), count in confusion_pairs.items():
        class_confusion_count[class1] = class_confusion_count.get(class1, 0) + count
        class_confusion_count[class2] = class_confusion_count.get(class2, 0) + count
    
    # Trier par nombre de confusions
    sorted_confusions = sorted(class_confusion_count.items(), 
                             key=lambda x: x[1], 
                             reverse=True)
    
    # Sauvegarder l'analyse en format texte
    analysis_path = f'confusion_analysis/{model_name}_confusions_{timestamp}.txt'
    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write(f"Analyse des confusions pour {model_name}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Classes les plus confondues:\n")
        for class_name, count in sorted_confusions[:10]:
            f.write(f"{class_name}: {str(count)} confusions\n")
        
        f.write("\nPaires de confusions spécifiques:\n")
        for (class1, class2), count in sorted(confusion_pairs.items(), 
                                            key=lambda x: x[1], 
                                            reverse=True)[:10]:
            f.write(f"{class1} <-> {class2}: {str(count)} fois\n")
    
    # Sauvegarder les données en CSV
    csv_path = f'confusion_analysis/csv/{model_name}_confusions_{timestamp}.csv'
    df = pd.DataFrame([(class1, class2, str(count)) for (class1, class2), count in confusion_pairs.items()],
                     columns=['class1', 'class2', 'count'])
    df.to_csv(csv_path, index=False)
    
    print(f"\nAnalyses sauvegardées dans:")
    print(f"- Texte: {analysis_path}")
    print(f"- CSV: {csv_path}")
    
    return sorted_confusions[0] if sorted_confusions else None

def save_and_visualize_results(results, benchmark_type="full", suffix=""):
    """Sauvegarde les résultats du benchmark"""
    try:
        # Création du dossier de résultats s'il n'existe pas
        os.makedirs('benchmark_results', exist_ok=True)
        
        # Ajout du timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Préparation des données pour le DataFrame
        data = []
        for model_name, metrics in results.items():
            row = {'Model': model_name}
            row.update(metrics)
            data.append(row)
        
        # Création du DataFrame
        df = pd.DataFrame(data)
        
        # Ajout de la colonne Timestamp
        df['Timestamp'] = timestamp
        
        # Sauvegarde en CSV
        csv_path = f'benchmark_results/benchmark_{benchmark_type}_{timestamp}{suffix}.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nDonnées traitées pour chaque modèle:")
        for row in data:
            print(row)
            
        print("\nDataFrame final:")
        print(df.drop('Timestamp', axis=1))
        print(f"\nRésultats sauvegardés dans: {csv_path}")
        
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des résultats: {str(e)}")
        traceback.print_exc()

def save_trained_model(model, model_name, metrics, save_dir="saved_models"):
    """Sauvegarde le modèle entraîné et ses métriques"""
    try:
        # Création du dossier de sauvegarde avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = Path(save_dir) / timestamp
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Nom du fichier avec informations importantes
        accuracy = metrics.get("Accuracy (%)", 0)
        model_file = save_path / f"{model_name}_acc{accuracy:.2f}_{timestamp}.pth"
        
        # Sauvegarde du modèle avec informations complètes
        save_dict = {
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
            'model_class': model.__class__.__name__,
            'metrics': metrics,
            'timestamp': timestamp,
            'architecture': str(model.__class__.__name__),
            'config': {
                'input_size': (224, 224),  # Taille standard pour la plupart des modèles
                'num_classes': 101  # Food-101 dataset
            }
        }
        
        # Sauvegarde avec gestion des erreurs
        torch.save(save_dict, model_file)
        
        # Vérification de la sauvegarde
        if model_file.exists():
            # Test de chargement
            try:
                checkpoint = torch.load(model_file)
                
                # Sauvegarde des métriques en format JSON pour référence facile
                metrics_file = save_path / f"{model_name}_metrics_{timestamp}.json"
                with open(metrics_file, 'w') as f:
                    json.dump({
                        'model_name': model_name,
                        'timestamp': timestamp,
                        'metrics': metrics,
                        'model_path': str(model_file)
                    }, f, indent=4)
                
                print(f"✓ Modèle sauvegardé: {model_file}")
                print(f"✓ Taille du fichier: {model_file.stat().st_size / (1024*1024):.2f} MB")
                print(f"✓ Métriques sauvegardées: {metrics_file}")
                
                return str(model_file)
            except Exception as load_error:
                print(f"⚠️ Erreur lors de la vérification du modèle: {load_error}")
                if model_file.exists():
                    model_file.unlink()  # Supprime le fichier corrompu
                return None
        else:
            print(f"⚠️ Erreur: Le fichier {model_file} n'a pas été créé")
            return None
            
    except Exception as e:
        print(f"⚠️ Erreur lors de la sauvegarde du modèle: {e}")
        traceback.print_exc()
        return None

def load_saved_model(model_path):
    """Charge un modèle sauvegardé avec toutes ses informations"""
    try:
        checkpoint = torch.load(model_path)
        
        # Récupération des informations du modèle
        model_name = checkpoint['model_name']
        model_class = checkpoint['model_class']
        
        # Création d'une nouvelle instance du modèle
        if model_name in models_to_test:
            model = models_to_test[model_name]
            
            # Chargement des poids
            model.load_state_dict(checkpoint['model_state_dict'])
            
            print(f"✓ Modèle {model_name} chargé avec succès")
            print(f"✓ Métriques originales: {checkpoint['metrics']}")
            print(f"✓ Date d'entraînement: {checkpoint['timestamp']}")
            
            return model, checkpoint['metrics']
        else:
            print(f"⚠️ Modèle {model_name} non trouvé dans la configuration")
            return None, None
            
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        traceback.print_exc()
        return None, None

def test_model_loading(model_path):
    """Teste le chargement d'un modèle sauvegardé"""
    try:
        print(f"\nTest de chargement du modèle: {model_path}")
        checkpoint = torch.load(model_path)
        print("✓ Modèle chargé avec succès")
        print(f"✓ Architecture: {checkpoint['architecture']}")
        print(f"✓ Timestamp: {checkpoint['timestamp']}")
        print(f"✓ Métriques: {checkpoint['metrics']}")
        return True
    except Exception as e:
        print(f"⚠️ Erreur lors du chargement du modèle: {e}")
        return False