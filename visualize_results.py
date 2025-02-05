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
        df_data.append({
            'Model': model_name,
            'Training Time (s)': metrics['Training Time (s)'],
            'Accuracy (%)': metrics['Accuracy (%)'],
            'Timestamp': timestamp,
            'Benchmark Type': benchmark_type
        })
    
    df = pd.DataFrame(df_data)
    
    # Sauvegarde des résultats dans un CSV
    csv_path = f'benchmark_results/benchmark_{benchmark_type}_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nRésultats sauvegardés dans: {csv_path}")
    
    # Création des visualisations
    plt.figure(figsize=(12, 5))
    
    # Graphique du temps d'entraînement
    plt.subplot(1, 2, 1)
    plt.bar(df['Model'], df['Training Time (s)'])
    plt.title('Temps d\'entraînement par modèle')
    plt.xticks(rotation=45)
    plt.ylabel('Temps (secondes)')
    
    # Graphique de la précision
    plt.subplot(1, 2, 2)
    plt.bar(df['Model'], df['Accuracy (%)'])
    plt.title('Précision par modèle')
    plt.xticks(rotation=45)
    plt.ylabel('Précision (%)')
    
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
        all_results = pd.concat([pd.read_csv(f'benchmark_results/{f}') for f in history_files])
        
        plt.figure(figsize=(12, 5))
        
        # Graphique de l'évolution des performances
        for model in all_results['Model'].unique():
            model_data = all_results[all_results['Model'] == model]
            plt.plot(model_data['Timestamp'], model_data['Accuracy (%)'], 
                    marker='o', label=model)
        
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