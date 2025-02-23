import os

def create_required_folders():
    """Crée tous les dossiers nécessaires pour le projet"""
    folders = [
        "api-test-endpoint",
        "saved_models",
        "data",
        "evaluation_results"
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✅ Dossier créé/vérifié: {folder}")

if __name__ == "__main__":
    create_required_folders() 