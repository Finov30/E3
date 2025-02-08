import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os

def save_and_visualize_results(results, benchmark_type="full"):
    # Création du dossier pour les résultats s'il n'existe pas
    os.makedirs('benchmark_results', exist_ok=True)
    
    # Préparation des données pour le DataFrame
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_data = []
    
    for model_name, metrics in results.items():
        try:
            # S'assurer que toutes les valeurs sont des nombres
            row_data = {
                'Model': str(model_name),
                'Training Time (s)': float(metrics.get('Training Time (s)', 0)),
                'Accuracy (%)': float(metrics.get('Accuracy (%)', 0)),
                'F1 Score (%)': float(metrics.get('F1 Score (%)', 0)),
                'Recall (%)': float(metrics.get('Recall (%)', 0)),
                'Timestamp': timestamp,
                'Benchmark Type': benchmark_type
            }
            df_data.append(row_data)
        except Exception as e:
            print(f"Erreur lors du traitement des métriques pour {model_name}: {e}")
            print(f"Métriques problématiques: {metrics}")
            # Ajouter une ligne avec des valeurs par défaut
            df_data.append({
                'Model': str(model_name),
                'Training Time (s)': 0.0,
                'Accuracy (%)': 0.0,
                'F1 Score (%)': 0.0,
                'Recall (%)': 0.0,
                'Timestamp': timestamp,
                'Benchmark Type': benchmark_type
            })
    
    df = pd.DataFrame(df_data)
    
    # Convertir explicitement les colonnes numériques
    numeric_columns = ['Training Time (s)', 'Accuracy (%)', 'F1 Score (%)', 'Recall (%)']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Sauvegarde des résultats dans un CSV
    csv_path = f'benchmark_results/benchmark_{benchmark_type}_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nRésultats sauvegardés dans: {csv_path}")
    
    # Création des visualisations
    plt.figure(figsize=(15, 10))
    
    # Graphique du temps d'entraînement
    plt.subplot(2, 2, 1)
    plt.bar(df['Model'], df['Training Time (s)'].astype(float))
    plt.title('Temps d\'entraînement par modèle')
    plt.xticks(rotation=45)
    plt.ylabel('Temps (secondes)')
    
    # Graphique des métriques de performance
    metrics_to_plot = ['Accuracy (%)', 'F1 Score (%)', 'Recall (%)']
    
    for idx, metric in enumerate(metrics_to_plot, 2):
        plt.subplot(2, 2, idx)
        plt.bar(df['Model'], df[metric].astype(float))
        plt.title(f'{metric} par modèle')
        plt.xticks(rotation=45)
        plt.ylabel(metric)
    
    plt.tight_layout()
    
    # Sauvegarde du graphique
    plot_path = f'benchmark_results/benchmark_plot_{benchmark_type}_{timestamp}.png'
    plt.savefig(plot_path)
    print(f"Visualisation sauvegardée dans: {plot_path}")
    
    # Affichage du graphique
    plt.show()
    
    visualize_history()

def visualize_history():
    # Chargement et affichage de l'historique des benchmarks
    history_files = [f for f in os.listdir('benchmark_results') if f.endswith('.csv')]
    if len(history_files) > 1:
        # Charger tous les fichiers CSV
        all_results = []
        for f in history_files:
            df = pd.read_csv(f'benchmark_results/{f}')
            # Convertir les colonnes numériques en float
            numeric_columns = ['Training Time (s)', 'Accuracy (%)', 'F1 Score (%)', 
                             'Recall (%)']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            all_results.append(df)
        
        all_results = pd.concat(all_results)
        
        plt.figure(figsize=(12, 5))
        
        # Graphique de l'évolution des performances
        for model in all_results['Model'].unique():
            model_data = all_results[all_results['Model'] == model].copy()
            # S'assurer que les valeurs sont numériques
            model_data['Accuracy (%)'] = pd.to_numeric(model_data['Accuracy (%)'], errors='coerce')
            plt.plot(model_data['Timestamp'], 
                    model_data['Accuracy (%)'], 
                    marker='o', 
                    label=model)
        
        plt.title('Évolution des performances au fil du temps')
        plt.xticks(rotation=45)
        plt.ylabel('Précision (%)')
        plt.legend()
        plt.tight_layout()
        
        # Sauvegarde du graphique d'historique
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_plot_path = f'benchmark_results/performance_history_{timestamp}.png'
        plt.savefig(history_plot_path)
        print(f"Historique des performances sauvegardé dans: {history_plot_path}")
        plt.show() 