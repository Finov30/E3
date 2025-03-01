#!/bin/bash

# Initialiser le modèle
python3 init_model.py

# Vérifier les variables d'environnement nécessaires
if [ -z "$MODEL_PATH" ]; then
    echo "❌ Erreur: MODEL_PATH non défini"
    exit 1
fi

# Vérifier si le modèle existe
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Erreur: Modèle non trouvé à l'emplacement: $MODEL_PATH"
    echo "Structure du répertoire saved_models:"
    ls -R /app/saved_models/
    echo "Chemin recherché: $MODEL_PATH"
    exit 1
fi

echo "✅ Modèle trouvé: $MODEL_PATH"
echo "📊 Taille du modèle: $(ls -lh $MODEL_PATH | awk '{print $5}')"

# Initialiser l'environnement avec les utilisateurs depuis les variables d'environnement
python3 -c "
from passlib.context import CryptContext
import json
import os

# Configuration du hachage
pwd_context = CryptContext(
    schemes=['sha256_crypt'],
    default='sha256_crypt',
    sha256_crypt__default_rounds=100000
)

# Créer les utilisateurs initiaux
users = {
    os.getenv('ADMIN_USERNAME', 'admin'): {
        'username': os.getenv('ADMIN_USERNAME', 'admin'),
        'password': pwd_context.hash(os.getenv('ADMIN_PASSWORD', 'admin123')),
        'role': 'admin',
        'api_keys': {}
    }
}

# Ajouter l'utilisateur par défaut si demandé
if os.getenv('CREATE_DEFAULT_USER', 'false').lower() == 'true':
    users[os.getenv('DEFAULT_USERNAME', 'user1')] = {
        'username': os.getenv('DEFAULT_USERNAME', 'user1'),
        'password': pwd_context.hash(os.getenv('DEFAULT_PASSWORD', 'user123')),
        'role': 'user',
        'api_keys': {}
    }

# Sauvegarder dans users.json
with open('users.json', 'w') as f:
    json.dump(users, f, indent=4)

print('✅ Utilisateurs initialisés avec succès')
"

# Vérifier CUDA
echo "🔍 Vérification de CUDA..."
python3 -c "
import torch
print('CUDA disponible:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Nombre de GPUs:', torch.cuda.device_count())
    print('GPU:', torch.cuda.get_device_name(0))
else:
    print('Mode CPU activé')
"

# Initialiser l'environnement de test
echo "🔧 Initialisation de l'environnement de test..."
python3 init_test_env.py

# Démarrer l'API en arrière-plan
python3 api.py &

# Attendre que l'API soit prête
echo "⏳ Attente du démarrage de l'API..."

# Fonction pour vérifier si l'API est prête
wait_for_api() {
    max_attempts=30
    attempt=1
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:8000/docs > /dev/null; then
            echo "✅ API prête!"
            return 0
        fi
        echo "Tentative $attempt/$max_attempts..."
        sleep 2
        attempt=$((attempt + 1))
    done
    echo "❌ L'API n'a pas démarré à temps"
    return 1
}

wait_for_api

# Exécuter les tests
echo "🧪 Exécution des tests..."
python3 test_endpoints.py

# Garder l'API en cours d'exécution
wait

python3 -m uvicorn api:app --host 0.0.0.0 --port 8000 