import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
import numpy as np

def save_and_visualize_results(results, benchmark_type="quick"):
    # Création du dossier pour les résultats s'il n'existe pas
    os.makedirs('benchmark_results', exist_ok=True)
    
    # Préparation des données pour le DataFrame
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_data = []
    
    # Debug: Afficher les résultats bruts
    print("\nRésultats bruts reçus:")
    for model_name, metrics in results.items():
        print(f"{model_name}: {metrics}")
    
    for model_name, metrics in results.items():
        try:
            row_data = {
                'Model': str(model_name),
                'Training Time (s)': float(metrics.get('Training Time (s)', 0)),
                'Accuracy (%)': float(metrics.get('Accuracy (%)', 0)),
                'F1 Score (%)': float(metrics.get('F1 Score (%)', 0)),
                'Recall (%)': float(metrics.get('Recall (%)', 0)),
                'Log Loss': float(metrics.get('Log Loss', 0)),
                'Top-3 Accuracy (%)': float(metrics.get('Top-3 Accuracy (%)', 0)),
                'Top-5 Accuracy (%)': float(metrics.get('Top-5 Accuracy (%)', 0))
            }
            df_data.append(row_data)
            
            # Debug: Afficher les données traitées
            print(f"\nDonnées traitées pour {model_name}:")
            print(row_data)
            
        except Exception as e:
            print(f"Erreur lors du traitement des métriques pour {model_name}: {e}")
            print(f"Métriques problématiques: {metrics}")
            df_data.append({
                'Model': str(model_name),
                'Training Time (s)': 0.0,
                'Accuracy (%)': 0.0,
                'F1 Score (%)': 0.0,
                'Recall (%)': 0.0,
                'Log Loss': 0.0,
                'Top-3 Accuracy (%)': 0.0,
                'Top-5 Accuracy (%)': 0.0
            })
    
    df = pd.DataFrame(df_data)
    
    # Debug: Afficher le DataFrame final
    print("\nDataFrame final:")
    print(df)
    
    # Création des visualisations
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Accuracy Comparison
    ax1 = fig.add_subplot(221)
    x = np.arange(len(df))
    width = 0.25
    
    ax1.bar(x - width, df['Accuracy (%)'], width, label='Top-1', color='blue')
    ax1.bar(x, df['Top-3 Accuracy (%)'], width, label='Top-3', color='green')
    ax1.bar(x + width, df['Top-5 Accuracy (%)'], width, label='Top-5', color='red')
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Model'], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Training Time
    ax2 = fig.add_subplot(222)
    ax2.bar(df['Model'], df['Training Time (s)'], color='purple')
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Time (s)')
    ax2.set_title('Training Time Comparison')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Log Loss
    ax3 = fig.add_subplot(223)
    log_loss_data = [metrics['Log Loss'] for metrics in results.values()]
    model_names = list(results.keys())
    ax3.bar(model_names, log_loss_data)
    ax3.set_title('Comparaison du Log Loss')
    ax3.set_ylabel('Log Loss')
    ax3.set_yscale('log')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    for i, v in enumerate(log_loss_data):
        ax3.text(i, v, f'{v:.2e}', ha='center', va='bottom')
    ax3.grid(True, alpha=0.3)
    
    # 4. F1 Score and Recall
    ax4 = fig.add_subplot(224)
    width = 0.35
    ax4.bar(x - width/2, df['F1 Score (%)'], width, label='F1 Score', color='green')
    ax4.bar(x + width/2, df['Recall (%)'], width, label='Recall', color='blue')
    ax4.set_xlabel('Models')
    ax4.set_ylabel('Score (%)')
    ax4.set_title('F1 Score and Recall Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(df['Model'], rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarde des résultats
    plot_path = f"benchmark_results/benchmark_{benchmark_type}_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Sauvegarde des résultats en CSV
    csv_path = f"benchmark_results/benchmark_{benchmark_type}_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"\nRésultats sauvegardés dans:")
    print(f"- Graphique: {plot_path}")
    print(f"- CSV: {csv_path}")
    
    # Visualisation de l'historique
    visualize_history()

def visualize_history():
    try:
        history_files = [f for f in os.listdir('benchmark_results') if f.endswith('.csv')]
        if len(history_files) > 1:
            all_results = []
            for f in history_files:
                try:
                    df = pd.read_csv(f'benchmark_results/{f}')
                    numeric_columns = ['Training Time (s)', 'Accuracy (%)', 'F1 Score (%)', 
                                     'Recall (%)', 'Log Loss', 'Top-3 Accuracy (%)', 'Top-5 Accuracy (%)']
                    for col in numeric_columns:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    all_results.append(df)
                except Exception as e:
                    print(f"Erreur lors de la lecture du fichier {f}: {e}")
            
            if all_results:
                all_results = pd.concat(all_results)
                
                plt.figure(figsize=(15, 8))
                for model in all_results['Model'].unique():
                    model_data = all_results[all_results['Model'] == model].copy()
                    plt.plot(model_data['Timestamp'], 
                            model_data['Accuracy (%)'], 
                            marker='o', 
                            label=model)
                
                plt.title('Évolution des performances au fil du temps')
                plt.xticks(rotation=45)
                plt.ylabel('Précision (%)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                history_plot_path = f'benchmark_results/performance_history_{timestamp}.png'
                plt.savefig(history_plot_path, dpi=300, bbox_inches='tight')
                print(f"Historique des performances sauvegardé dans: {history_plot_path}")
                plt.close()
    except Exception as e:
        print(f"Erreur lors de la visualisation de l'historique: {e}") 