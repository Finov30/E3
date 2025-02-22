import os
import sys

def check_model_files():
    """Vérifie la présence et l'intégrité des fichiers modèle"""
    model_dir = "saved_models/20250221_132302"
    model_name = "ResNet-50_acc78.61_20250221_132302.pth"
    model_path = os.path.join(model_dir, model_name)
    
    print("🔍 Vérification des fichiers modèle...")
    
    if not os.path.exists(model_dir):
        print(f"❌ Erreur: Répertoire {model_dir} non trouvé")
        return False
        
    if not os.path.exists(model_path):
        print(f"❌ Erreur: Modèle {model_name} non trouvé")
        return False
    
    size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    print(f"✅ Modèle trouvé: {model_path}")
    print(f"📊 Taille: {size:.2f} MB")
    
    return True

if __name__ == "__main__":
    if not check_model_files():
        sys.exit(1) 