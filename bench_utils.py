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
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from bench_config import OPTIMIZER_CONFIG
from torchvision.datasets import Food101
from datetime import datetime
import pandas as pd
import traceback
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
import warnings

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
        
        return training_time
        
    except Exception as e:
        print(f"Erreur dans train_model: {e}")
        raise
    finally:
        # S'assurer que la session est fermée
        monitor.close()
        if mlflow.active_run():
            mlflow.end_run()

def get_model_colormap(model_name):
    """Retourne une colormap spécifique pour chaque modèle"""
    if "ResNet" in model_name:
        # Palette de bleus pour ResNet
        colors = [
            "#FFFFFF",  # Blanc pour 0
            "#E6F3FF",  # Bleu très clair
            "#B3D9FF",  # Bleu clair
            "#80C0FF",  # Bleu moyen
            "#4DA6FF",  # Bleu vif
            "#1A8CFF",  # Bleu intense
            "#0073E6",  # Bleu profond
            "#005CB3",  # Bleu foncé
            "#004080",  # Bleu très foncé
            "#00264D"   # Bleu nuit
        ]
    elif "EfficientNet" in model_name:
        # Palette de verts pour EfficientNet
        colors = [
            "#FFFFFF",  # Blanc pour 0
            "#E6FFE6",  # Vert très clair
            "#B3FFB3",  # Vert clair
            "#80FF80",  # Vert moyen
            "#4DFF4D",  # Vert vif
            "#1AFF1A",  # Vert intense
            "#00E600",  # Vert profond
            "#00B300",  # Vert foncé
            "#008000",  # Vert très foncé
            "#004D00"   # Vert forêt
        ]
    else:  # MobileNetV3 et autres
        # Palette de violets pour MobileNet
        colors = [
            "#FFFFFF",  # Blanc pour 0
            "#F3E6FF",  # Violet très clair
            "#D9B3FF",  # Violet clair
            "#C080FF",  # Violet moyen
            "#A64DFF",  # Violet vif
            "#8C1AFF",  # Violet intense
            "#7300E6",  # Violet profond
            "#5C00B3",  # Violet foncé
            "#400080",  # Violet très foncé
            "#26004D"   # Violet nuit
        ]
    return LinearSegmentedColormap.from_list("custom", colors)

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
        running_log_loss = 0.0
        top_k_correct = {1: 0, 3: 0, 5: 0}
        
        criterion = nn.CrossEntropyLoss(reduction='sum')
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Evaluation')
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                
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
                all_probs.extend(probabilities.cpu().numpy())
                
                accuracy = 100 * correct / total
                metrics = {
                    "batch_accuracy": accuracy,
                    "batch_size": labels.size(0)
                }
                monitor.log_batch_metrics(metrics, step=batch_idx)
                pbar.set_postfix({'accuracy': f'{accuracy:.2f}%'})
        
        # Calcul des métriques finales
        final_accuracy = 100 * correct / total
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        print(f"Debug - Precision: {precision:.4f}")
        print(f"Debug - Recall: {recall:.4f}")
        print(f"Debug - F1 calculated: {f1:.4f}")
        print(f"Debug - F1 from P&R: {2 * (precision * recall) / (precision + recall):.4f}")
        
        log_loss = running_log_loss / total
        top_k_accuracy = {k: (100 * v / total) for k, v in top_k_correct.items()}
        
        # Création et sauvegarde de la matrice de confusion
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        
        # Récupération des noms de classes depuis le fichier classes.txt
        with open('data/food-101/meta/classes.txt', 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
            
        # Obtenir la colormap spécifique au modèle
        custom_cmap = get_model_colormap(model_name)
        
        # Ajuster la normalisation pour une meilleure visualisation
        # Utiliser une normalisation logarithmique avec un seuil minimum ajusté
        min_value = max(0.1, conf_matrix.min())
        norm = LogNorm(vmin=min_value, vmax=conf_matrix.max())
        
        # Création de la heatmap avec la nouvelle colormap
        plt.figure(figsize=(30, 30))
        plt.rcParams.update({'font.size': 6})
        
        # Améliorer la lisibilité des annotations
        sns.heatmap(conf_matrix,
                    annot=True,
                    fmt='d',
                    cmap=custom_cmap,
                    norm=norm,
                    xticklabels=class_names,
                    yticklabels=class_names,
                    square=True,
                    cbar_kws={'label': 'Nombre de prédictions'},
                    annot_kws={'size': 5, 'weight': 'bold'})
        
        # Amélioration du style et de la lisibilité
        plt.title(f'Matrice de Confusion - {model_name}',
                 pad=20,
                 fontsize=16,
                 fontweight='bold')
        
        # Ajouter un fond plus clair pour les labels
        plt.xlabel('Prédictions', labelpad=10, fontsize=12, fontweight='bold')
        plt.ylabel('Vraies Classes', labelpad=10, fontsize=12, fontweight='bold')
        
        # Améliorer la lisibilité des labels
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=45)
        
        # Ajouter une grille fine pour mieux distinguer les cellules
        plt.grid(True, which='minor', color='gray', linewidth=0.1, alpha=0.2)
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        # Création du dossier pour les matrices de confusion
        confusion_matrix_dir = 'benchmark_results/confusion_matrices'
        os.makedirs(confusion_matrix_dir, exist_ok=True)
        
        # Sauvegarde avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sauvegarder la matrice en format PNG haute résolution
        confusion_matrix_path = f'{confusion_matrix_dir}/{model_name}_confusion_matrix_{timestamp}.png'
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Sauvegarder la matrice en format CSV
        confusion_matrix_csv = f'{confusion_matrix_dir}/{model_name}_confusion_matrix_{timestamp}.csv'
        confusion_df = pd.DataFrame(conf_matrix, 
                                  index=class_names,
                                  columns=class_names)
        confusion_df.to_csv(confusion_matrix_csv)
        
        # Analyse des confusions les plus fréquentes
        confusion_pairs = {}
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i != j and conf_matrix[i][j] > 0:
                    pair = (class_names[i], class_names[j])
                    confusion_pairs[pair] = conf_matrix[i][j]
        
        # Trier les confusions par fréquence
        sorted_confusions = sorted(confusion_pairs.items(), 
                                 key=lambda x: x[1], 
                                 reverse=True)
        
        # Sauvegarder l'analyse des confusions
        confusion_analysis_dir = 'confusion_analysis'
        os.makedirs(confusion_analysis_dir, exist_ok=True)
        
        analysis_path = f'{confusion_analysis_dir}/{model_name}_confusions_{timestamp}.txt'
        with open(analysis_path, 'w', encoding='utf-8') as f:
            f.write(f"Analyse des confusions pour {model_name}\n")
            f.write("=" * 50 + "\n\n")
            
            # Top 10 des classes les plus confondues
            class_confusion_count = {}
            for (class1, class2), count in confusion_pairs.items():
                class_confusion_count[class1] = class_confusion_count.get(class1, 0) + count
                class_confusion_count[class2] = class_confusion_count.get(class2, 0) + count
            
            sorted_classes = sorted(class_confusion_count.items(), 
                                  key=lambda x: x[1], 
                                  reverse=True)
            
            f.write("Classes les plus confondues:\n")
            for class_name, count in sorted_classes[:10]:
                f.write(f"{class_name}: {count} confusions\n")
            
            f.write("\nPaires de confusions spécifiques:\n")
            for (class1, class2), count in sorted_confusions[:10]:
                f.write(f"{class1} <-> {class2}: {count} fois\n")
        
        print(f"\nMatrices de confusion sauvegardées:")
        print(f"- Image: {confusion_matrix_path}")
        print(f"- CSV: {confusion_matrix_csv}")
        print(f"- Analyse: {analysis_path}")
        
        final_metrics = {
            "test_accuracy": float(final_accuracy),
            "f1_score": float(f1 * 100),
            "recall_score": float(recall * 100),
            "log_loss": float(log_loss),
            "top_1_accuracy": float(top_k_accuracy[1]),
            "top_3_accuracy": float(top_k_accuracy[3]),
            "top_5_accuracy": float(top_k_accuracy[5]),
            "total_samples": int(total),
            "model_name": str(model_name or model.__class__.__name__)
        }
        
        if monitor.mlflow_active:
            mlflow.log_metrics({k: v for k, v in final_metrics.items() 
                              if isinstance(v, (int, float))})
            mlflow.log_artifact(confusion_matrix_path)
        
        return final_metrics
        
    except Exception as e:
        print(f"\nUne erreur s'est produite: {str(e)}")
        traceback.print_exc()
        raise
    finally:
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

def analyze_similar_classes(model, test_loader, device, model_name, n_similar=5):
    """Analyse les classes souvent confondues entre elles"""
    model.eval()
    all_predictions = []
    all_labels = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Récupérer les classes depuis le dataset original
    dataset = Food101(root='./data', split='test', download=True)
    class_names = dataset.classes
    
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

def analyze_and_save_confusions(model_name, confusion_pairs, timestamp):
    """Analyse et sauvegarde les confusions fréquentes"""
    # Création des dossiers pour les analyses
    os.makedirs('confusion_analysis', exist_ok=True)
    os.makedirs('confusion_analysis/csv', exist_ok=True)
    os.makedirs('benchmark_results/confusion_matrices', exist_ok=True)
    
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
    
    # Sauvegarder le résumé par classe en CSV
    summary_csv_path = f'confusion_analysis/csv/{model_name}_confusion_summary_{timestamp}.csv'
    df_summary = pd.DataFrame([(class_name, str(count)) for class_name, count in sorted_confusions],
                            columns=['class_name', 'total_confusions'])
    df_summary.to_csv(summary_csv_path, index=False)
    
    print(f"\nAnalyses sauvegardées dans:")
    print(f"- Texte: {analysis_path}")
    print(f"- CSV paires: {csv_path}")
    print(f"- CSV résumé: {summary_csv_path}")
    
    return sorted_confusions[0] if sorted_confusions else None

def validate_transforms(image):
    try:
        transformed = data_transforms(image)
        return True
    except Exception as e:
        print(f"Erreur de transformation: {e}")
        return False

def save_and_visualize_results(results, benchmark_type="full", suffix=""):
    """Sauvegarde et visualise les résultats du benchmark"""
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
        
        # Création des visualisations
        plt.figure(figsize=(15, 10))
        
        # Graphique des métriques principales
        metrics_to_plot = ['Accuracy (%)', 'F1 Score (%)', 'Top-3 Accuracy (%)', 'Top-5 Accuracy (%)']
        bar_width = 0.2
        x = np.arange(len(df['Model']))
        
        for i, metric in enumerate(metrics_to_plot):
            plt.bar(x + i*bar_width, df[metric], bar_width, label=metric)
        
        plt.xlabel('Modèles')
        plt.ylabel('Score (%)')
        plt.title('Comparaison des performances des modèles')
        plt.xticks(x + bar_width*1.5, df['Model'], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Sauvegarde du graphique
        plot_path = f'benchmark_results/benchmark_{benchmark_type}_{timestamp}{suffix}.png'
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"\nRésultats sauvegardés dans:")
        print(f"- Graphique: {plot_path}")
        print(f"- CSV: {csv_path}")
        
        # Visualisation de l'historique si disponible
        try:
            history_files = sorted(
                [f for f in os.listdir('benchmark_results') 
                 if f.startswith(f'benchmark_{benchmark_type}') and f.endswith('.csv')]
            )
            
            if len(history_files) > 1:
                history_data = []
                for file in history_files:
                    hist_df = pd.read_csv(f'benchmark_results/{file}')
                    if 'Timestamp' not in hist_df.columns:
                        hist_df['Timestamp'] = file.split('_')[2].split('.')[0]
                    history_data.append(hist_df)
                
                history_df = pd.concat(history_data, ignore_index=True)
                
                # Création du graphique d'historique
                plt.figure(figsize=(15, 10))
                for model in history_df['Model'].unique():
                    model_data = history_df[history_df['Model'] == model]
                    plt.plot(model_data['Timestamp'], model_data['Accuracy (%)'], 
                            marker='o', label=model)
                
                plt.xlabel('Date')
                plt.ylabel('Accuracy (%)')
                plt.title('Évolution de la précision au fil du temps')
                plt.xticks(rotation=45)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Sauvegarde de l'historique
                history_path = f'benchmark_results/history_{benchmark_type}_{timestamp}.png'
                plt.savefig(history_path, bbox_inches='tight', dpi=300)
                plt.close()
                print(f"- Historique: {history_path}")
                
        except Exception as e:
            print(f"Erreur lors de la visualisation de l'historique: {str(e)}")
            
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des résultats: {str(e)}")
        traceback.print_exc()