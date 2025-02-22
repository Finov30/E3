import json
from passlib.context import CryptContext
import os
from dotenv import load_dotenv

def init_environment():
    """Initialise l'environnement avec les configurations de base"""
    load_dotenv()
    
    # Vérifier si l'environnement est déjà initialisé
    if os.path.exists("users.json"):
        print("⚠️ L'environnement est déjà initialisé!")
        return False
    
    # Configuration du hachage
    pwd_context = CryptContext(
        schemes=["sha256_crypt"],
        default="sha256_crypt",
        sha256_crypt__default_rounds=100000,
        deprecated="auto"
    )
    
    # Demander les informations d'admin
    print("Configuration de l'administrateur:")
    admin_username = input("Nom d'utilisateur admin (défaut 'admin'): ") or "admin"
    admin_password = input("Mot de passe admin: ")
    
    # Créer la structure initiale des utilisateurs avec mot de passe hashé
    initial_users = {
        admin_username: {
            "username": admin_username,
            "password": pwd_context.hash(admin_password),
            "role": "admin",
            "api_keys": {}
        }
    }
    
    # Sauvegarder la configuration
    try:
        with open("users.json", 'w') as f:
            json.dump(initial_users, f, indent=4)
        
        print("\n✅ Environnement initialisé avec succès!")
        print(f"Username: {admin_username}")
        print("N'oubliez pas de sauvegarder ces informations de manière sécurisée.")
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur lors de l'initialisation: {str(e)}")
        return False

if __name__ == "__main__":
    init_environment() 