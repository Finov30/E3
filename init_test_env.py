import os

def init_test_environment():
    """Initialise l'environnement de test"""
    test_dir = "api-test-endpoint"
    os.makedirs(test_dir, exist_ok=True)
    
    print(f"📂 Contenu du dossier de travail actuel:")
    print(os.listdir("."))
    
    print(f"\n📂 Contenu du dossier {test_dir}:")
    if os.path.exists(test_dir):
        print(os.listdir(test_dir))
    else:
        print("Le dossier n'existe pas!")
    
    required_files = {
        "images-test.jpg": "Image de test unique",
        "zip-test.zip": "Archive ZIP de test",
        "multi-image-test.jpg": "Première image pour test batch",
        "multi-image-test-2.jpg": "Deuxième image pour test batch"
    }
    
    for filename, description in required_files.items():
        file_path = os.path.join(test_dir, filename)
        if not os.path.exists(file_path):
            print(f"❌ Erreur: {description} non trouvé: {filename}")
            print(f"   Chemin complet: {os.path.abspath(file_path)}")
            return False
        else:
            print(f"✅ Trouvé: {filename}")
    
    print("✅ Tous les fichiers de test vérifiés avec succès")
    return True

if __name__ == "__main__":
    init_test_environment() 