import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from mlflow_manager import MLflowManager

def print_banner():
    print("""
╔══════════════════════════════════════╗
║         Food-101 Benchmark           ║
╚══════════════════════════════════════╝
    """)

def get_user_choice():
    while True:
        print("\nChoisissez le type de benchmark à exécuter:")
        print("1. Benchmark rapide (petit échantillon)")
        print("2. Benchmark complet (dataset entier)")
        print("q. Quitter")
        
        choice = input("\nVotre choix (1, 2 ou q) : ").lower()
        
        if choice in ['1', '2', 'q']:
            return choice
        else:
            print("\nChoix invalide. Veuillez réessayer.")

def main():
    print_banner()
    
    # Initialisation de MLflow
    mlflow_manager = MLflowManager()
    if not mlflow_manager.initialize():
        print("⚠️ Attention: MLflow n'a pas pu être initialisé. Le monitoring sera limité.")
    
    while True:
        choice = get_user_choice()
        
        if choice == 'q':
            print("\nAu revoir!")
            break
        
        try:
            if choice == '1':
                print("\nLancement du benchmark rapide...")
                import bench_quick
            else:
                print("\nLancement du benchmark complet...")
                import bench
                
        except Exception as e:
            print(f"\nUne erreur s'est produite: {str(e)}")
            
        input("\nAppuyez sur Entrée pour continuer...")

if __name__ == "__main__":
    main() 