import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from mlflow_manager import MLflowManager

def print_banner():
    print("""
╔══════════════════════════════════════╗
║         Food-101 Benchmark           ║
║           ResNet-50 Only             ║
╚══════════════════════════════════════╝
    """)

def get_user_choice():
    while True:
        print("\nAppuyez sur:")
        print("1. Lancer le benchmark complet")
        print("q. Quitter")
        
        choice = input("\nVotre choix (1 ou q) : ").lower()
        
        if choice in ['1', 'q']:
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
            print("\nLancement du benchmark complet...")
            import bench
                
        except Exception as e:
            print(f"\nUne erreur s'est produite: {str(e)}")
            
        input("\nAppuyez sur Entrée pour continuer...")

if __name__ == "__main__":
    main() 