from fastapi import FastAPI, UploadFile, File, HTTPException, Security, Depends, Form, Query
from fastapi.responses import JSONResponse
from starlette.status import HTTP_403_FORBIDDEN
import torch
import torch.nn as nn
from bench_config import transform, device, models_to_test
from PIL import Image
import io
import uvicorn
from pathlib import Path
import logging
from typing import List, Optional, Dict
import zipfile
import tempfile
import os
from torchvision.datasets import Food101
from dotenv import load_dotenv
from datetime import datetime, timedelta
import secrets
import json
from cryptography.fernet import Fernet
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from passlib.context import CryptContext
from contextlib import asynccontextmanager
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import jwt
from jose import JWTError
from collections import deque
import torch.optim as optim
import mlflow
from torch.utils.data import TensorDataset, DataLoader
import uuid
import subprocess
import time
import requests
import copy
from fastapi.testclient import TestClient

# Chargement des variables d'environnement
load_dotenv()

# Configuration des constantes
FEEDBACK_THRESHOLD = 10
CACHE_TTL_MINUTES = 30

# Définition des caches globaux
predictions_cache = {}
feedback_cache = []

class PredictionCache:
    def __init__(self):
        self.cache = {}
        self.ttl = timedelta(minutes=CACHE_TTL_MINUTES)
    
    def add(self, prediction_id: str, data: dict):
        self.cache[prediction_id] = {
            "data": data,
            "timestamp": datetime.now()
        }
    
    def get(self, prediction_id: str) -> Optional[dict]:
        if prediction_id not in self.cache:
            return None
        
        entry = self.cache[prediction_id]
        if datetime.now() - entry["timestamp"] > self.ttl:
            del self.cache[prediction_id]
            return None
            
        return entry["data"]

# Initialiser les caches
predictions_cache = PredictionCache()
feedback_cache = deque(maxlen=FEEDBACK_THRESHOLD)

# Déclarations globales au début du fichier
model = None
device = None
continuous_learning_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application"""
    try:
        global model, device, continuous_learning_manager
        
        # 1. Initialisation des logs
        logger.info("🚀 Démarrage de l'API Food101...")
        
        # 2. Vérification des dossiers requis
        required_dirs = [
            "/app/mlruns",
            "/app/artifacts",
            "/app/artifacts/model_backups",
            "/app/artifacts/training_summaries",
            "/app/data",
            "/app/saved_models"
        ]
        
        for directory in required_dirs:
            try:
                os.makedirs(directory, exist_ok=True)
                os.chmod(directory, 0o777)
                logger.info(f"✅ Dossier créé/vérifié: {directory}")
            except Exception as e:
                logger.error(f"❌ Erreur création dossier {directory}: {str(e)}")
                raise
        
        # 3. Configuration de MLflow
        if not setup_mlflow():
            logger.error("❌ Configuration MLflow échouée")
            raise Exception("Échec de la configuration MLflow")
        logger.info("✅ MLflow configuré")
        
        # 4. Chargement du modèle
        if not load_model():
            logger.error("❌ Chargement du modèle échoué")
            raise Exception("Échec du chargement du modèle")
        logger.info("✅ Modèle chargé")
        
        # 5. Initialisation du gestionnaire d'apprentissage continu
        try:
            continuous_learning_manager = ContinuousLearningManager(model, device)
            logger.info("✅ Gestionnaire d'apprentissage continu initialisé")
        except Exception as e:
            logger.error(f"❌ Erreur initialisation apprentissage continu: {str(e)}")
            raise
        
        # 6. Test initial d'apprentissage continu
        try:
            await test_continuous_learning()
            logger.info("✅ Test d'apprentissage continu terminé")
        except Exception as e:
            logger.error(f"⚠️ Test d'apprentissage continu échoué: {str(e)}")
            # Ne pas bloquer le démarrage si le test échoue
        
        logger.info("✅ API prête à recevoir des requêtes!")
        yield
        
    except Exception as e:
        logger.error(f"❌ Erreur fatale lors de l'initialisation: {str(e)}")
        raise
        
    finally:
        # Nettoyage à l'arrêt
        logger.info("Arrêt de l'API...")
        if hasattr(app.state, 'mlflow_process'):
            logger.info("Arrêt du serveur MLflow...")
            app.state.mlflow_process.terminate()
            app.state.mlflow_process.wait()
            logger.info("✅ Serveur MLflow arrêté")
        
        if model is not None:
            model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("✅ Ressources modèle libérées")

app = FastAPI(
    title="Food101 Classification API",
    description="API pour classifier des images de nourriture parmi 101 classes",
    version="1.0.0",
    lifespan=lifespan
)

# Configuration OAuth2 simplifiée
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="auth/login",
    auto_error=True
)

def create_access_token(data: dict, expires_delta: timedelta = timedelta(days=1)):
    """Crée un token JWT avec les données utilisateur"""
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, os.getenv("SECRET_KEY"), algorithm="HS256")
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Vérifie le token JWT et retourne l'utilisateur"""
    # Autoriser l'utilisateur système pour les tests
    if token == "system_test":
        return "system_test"
        
    try:
        payload = jwt.decode(token, os.getenv("SECRET_KEY"), algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None or username not in api_manager.users:
            raise HTTPException(status_code=401)
        return username
    except JWTError:
        raise HTTPException(status_code=401)

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration du hachage et du cryptage
class PasswordSecurityManager:
    def __init__(self):
        # Configuration du hachage avec paramètres personnalisés
        self.pwd_context = CryptContext(
            schemes=["sha256_crypt"],  # Utiliser sha256 au lieu de bcrypt
            default="sha256_crypt",
            sha256_crypt__default_rounds=100000,  # Nombre d'itérations
            deprecated="auto"
        )
        
        # Configuration du cryptage
        secret_key = os.getenv("SECRET_KEY", "your-secret-key").encode()
        salt = b"fixed_salt"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(secret_key))
        self.cipher_suite = Fernet(key)

    def secure_password(self, password: str) -> str:
        """Hache le mot de passe"""
        return self.pwd_context.hash(password)

    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Vérifie un mot de passe hashé"""
        try:
            return self.pwd_context.verify(password, hashed_password)
        except Exception as e:
            logger.error(f"Erreur lors de la vérification du mot de passe: {str(e)}")
            return False

# Initialiser le gestionnaire de sécurité
password_security = PasswordSecurityManager()

# Classe pour gérer les utilisateurs et leurs clés API
class APIKeyManager:
    def __init__(self, users_file: str = "users.json"):
        self.users_file = users_file
        self.users = self._load_users()
        self.security_manager = PasswordSecurityManager()

    def _load_users(self) -> Dict:
        """Charge les utilisateurs depuis le fichier JSON"""
        try:
            with open(self.users_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise Exception("Environment not initialized. Please run init_environment.py first")

    def _save_users(self, users: Dict = None):
        """Sauvegarde les utilisateurs dans le fichier JSON"""
        with open(self.users_file, 'w') as f:
            json.dump(users or self.users, f, indent=4)

    def generate_api_key(self, username: str, expiration_days: int = 30) -> Optional[str]:
        """Génère une nouvelle clé API pour un utilisateur"""
        if username not in self.users:
            return None
        
        # Générer une nouvelle clé
        api_key = secrets.token_urlsafe(32)
        expiration = (datetime.now() + timedelta(days=expiration_days)).isoformat()
        
        # Sauvegarder la clé
        self.users[username]["api_keys"][api_key] = {
            "created_at": datetime.now().isoformat(),
            "expires_at": expiration
        }
        self._save_users()
        
        return api_key

    def revoke_api_key(self, username: str, api_key: str) -> bool:
        """Révoque une clé API"""
        if username not in self.users or api_key not in self.users[username]["api_keys"]:
            return False
        
        del self.users[username]["api_keys"][api_key]
        self._save_users()
        return True

    def is_valid_api_key(self, api_key: str) -> bool:
        """Vérifie si une clé API est valide"""
        for user in self.users.values():
            if api_key in user["api_keys"]:
                expires_at = datetime.fromisoformat(user["api_keys"][api_key]["expires_at"])
                return datetime.now() < expires_at
        return False

    def get_user_keys(self, username: str) -> Optional[Dict]:
        """Récupère toutes les clés API d'un utilisateur"""
        if username not in self.users:
            return None
        return self.users[username]["api_keys"]

    def verify_password(self, username: str, password: str) -> bool:
        """Vérifie le mot de passe d'un utilisateur"""
        try:
            if username not in self.users:
                return False
            
            stored_password = self.users[username]["password"]
            # Si le mot de passe n'est pas hashé (première connexion)
            if not stored_password.startswith("$"):  # Changé de $2b$ à $
                if password == stored_password:  # Comparaison directe pour la première fois
                    # Hasher le mot de passe pour les prochaines fois
                    self.users[username]["password"] = self.security_manager.secure_password(password)
                    self._save_users()
                    return True
                return False
            
            # Vérifier le mot de passe hashé
            return self.security_manager.verify_password(password, stored_password)
        except Exception as e:
            logger.error(f"Erreur lors de la vérification du mot de passe: {str(e)}")
            return False

    def create_user(self, username: str, password: str, role: str = "user") -> bool:
        """Crée un nouvel utilisateur"""
        if username in self.users:
            return False
        
        self.users[username] = {
            "username": username,
            "password": self.security_manager.secure_password(password),
            "role": role,
            "api_keys": {}
        }
        self._save_users()
        return True

    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """Change le mot de passe d'un utilisateur"""
        if not self.verify_password(username, old_password):
            return False
        
        self.users[username]["password"] = self.security_manager.secure_password(new_password)
        self._save_users()
        return True

# Initialisation du gestionnaire de clés API
api_manager = APIKeyManager()

# Ajouter après la création de l'app FastAPI
client = TestClient(app)

# Après les imports et avant les variables globales
class ContinuousLearningManager:
    def __init__(self, base_model, device, buffer_size=100, min_feedback=10):
        # Modèle de base pour les prédictions
        self.base_model = base_model
        # Modèle d'entraînement initialisé avec le modèle de base
        self.training_model = copy.deepcopy(base_model)
        
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.feedback_buffer = deque(maxlen=buffer_size)
        self.min_feedback_count = min_feedback
        
        # Chemins pour les sauvegardes
        self.backup_dir = "/app/artifacts/model_backups"
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Initialiser les classes
        self._init_classes()
        
        # Charger la dernière version d'entraînement si elle existe
        self.version = self._get_latest_version()
        if self.version > 0:
            self._load_latest_training_model()
        
        # Optimiser uniquement la dernière couche
        for param in self.training_model.parameters():
            param.requires_grad = False
        self.training_model.fc.requires_grad = True
        self.optimizer = optim.SGD(self.training_model.fc.parameters(), lr=0.0001, momentum=0.9)

    def _init_classes(self):
        """Initialise la liste des classes"""
        try:
            classes_path = "./data/food-101/meta/classes.txt"
            if os.path.exists(classes_path):
                with open(classes_path, 'r') as f:
                    self.classes = [line.strip() for line in f.readlines()]
                logger.info(f"✅ {len(self.classes)} classes chargées depuis {classes_path}")
            else:
                dataset = Food101(root='./data', split='train', download=True)
                self.classes = dataset.classes
                logger.info("✅ Classes chargées depuis le dataset Food101")
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation des classes: {str(e)}")
            raise

    def _get_latest_version(self):
        """Récupère la dernière version du modèle"""
        try:
            versions = [int(f.split('v')[1].split('_')[0]) 
                       for f in os.listdir(self.backup_dir) 
                       if f.startswith('v') and '_model.pth' in f]
            return max(versions) if versions else 0
        except Exception:
            return 0

    def _save_model_version(self, metrics):
        """Sauvegarde une nouvelle version du modèle"""
        try:
            self.version += 1
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            
            # Nom des fichiers
            model_filename = f"v{self.version}_model_{timestamp}.pth"
            metadata_filename = f"v{self.version}_metadata.json"
            
            # Chemins complets
            model_path = os.path.join(self.backup_dir, model_filename)
            metadata_path = os.path.join(self.backup_dir, metadata_filename)
            
            # Sauvegarder le modèle
            save_data = {
                'model_state_dict': self.training_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'version': self.version,
                'timestamp': timestamp,
                'metrics': metrics
            }
            torch.save(save_data, model_path)
            
            # Sauvegarder les métadonnées
            metadata = {
                'version': self.version,
                'timestamp': timestamp,
                'metrics': metrics,
                'model_file': model_filename
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
                
            logger.info(f"✅ Modèle v{self.version} sauvegardé: {model_filename}")
            logger.info(f"📊 Métadonnées sauvegardées: {metadata_filename}")
            
            return model_path
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la sauvegarde du modèle v{self.version}: {str(e)}")
            raise

    def _load_latest_training_model(self):
        """Charge la dernière version du modèle d'entraînement"""
        try:
            # Trouver le dernier fichier de modèle
            model_files = sorted([f for f in os.listdir(self.backup_dir) 
                                if f.startswith(f'v{self.version}_') and f.endswith('.pth')])
            if model_files:
                latest_model = model_files[-1]
                model_path = os.path.join(self.backup_dir, latest_model)
                logger.info(f"🔄 Chargement du modèle d'entraînement: {latest_model}")
                
                # Charger l'état du modèle
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Préparer le modèle d'entraînement
                self.training_model = prepare_model(self.training_model)
                
                # Charger les poids
                if "model_state_dict" in checkpoint:
                    self.training_model.load_state_dict(checkpoint["model_state_dict"])
                    if "optimizer_state_dict" in checkpoint:
                        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                else:
                    self.training_model.load_state_dict(checkpoint)
                
                logger.info("✅ Modèle d'entraînement chargé avec succès")
                
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle d'entraînement: {str(e)}")
            # En cas d'erreur, on garde le modèle de base
            self.training_model = copy.deepcopy(self.base_model)
            logger.info("⚠️ Utilisation du modèle de base comme modèle d'entraînement")

    async def train_on_feedback(self, feedbacks):
        """Entraîne le modèle sur les feedbacks"""
        logger.info(f"\n🚀 Démarrage de l'entraînement v{self.version + 1}")
        
        try:
            # Créer un nom unique pour ce run
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            run_name = f"training_v{self.version + 1}_{timestamp}"
            
            # Vérifier que MLflow est accessible
            try:
                mlflow.get_tracking_uri()
            except Exception as e:
                logger.error(f"❌ MLflow n'est pas accessible: {str(e)}")
                raise
            
            with mlflow.start_run(run_name=run_name) as run:
                # Log des paramètres avec vérification
                params = {
                    "version": self.version + 1,
                    "feedback_count": len(feedbacks),
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                    "momentum": 0.9,
                    "timestamp": timestamp
                }
                mlflow.log_params(params)
                logger.info("✅ Paramètres enregistrés dans MLflow")
                
                # Préparation des données
                train_tensors = []
                train_labels = []
                
                for feedback in feedbacks:
                    # Corriger la forme du tenseur (4D)
                    image_tensor = feedback["image_tensor"].squeeze(0)  # Enlever la dimension batch inutile
                    image_tensor = image_tensor.requires_grad_(True)  # Activer les gradients
                    
                    correct_class = feedback["used_class"]
                    label = self.classes.index(correct_class)
                    
                    train_tensors.append(image_tensor)
                    train_labels.append(label)
                
                # Création du dataset avec les bonnes dimensions
                train_tensors = torch.stack(train_tensors)  # [N, C, H, W]
                train_labels = torch.tensor(train_labels, dtype=torch.long)
                dataset = TensorDataset(train_tensors, train_labels)
                dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
                
                # Entraînement
                self.training_model.train()
                total_loss = 0
                correct = 0
                total = 0
                
                for batch_idx, (inputs, targets) in enumerate(dataloader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    self.optimizer.zero_grad()
                    outputs = self.training_model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    
                    # Log des métriques par batch
                    mlflow.log_metrics({
                        "batch_loss": loss.item(),
                        "batch_accuracy": 100 * correct / total
                    }, step=batch_idx)
                
                # Calculer les métriques finales
                metrics = {
                    "avg_loss": total_loss / len(dataloader),
                    "accuracy": 100 * correct / total,
                    "feedback_count": len(feedbacks)
                }
                
                # Sauvegarder le modèle avec vérification
                model_filename = f"model_v{self.version + 1}_{timestamp}.pth"
                model_path = os.path.join("/app/saved_models", model_filename)
                
                # Sauvegarder localement d'abord
                save_data = {
                    'model_state_dict': self.training_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'version': self.version + 1,
                    'metrics': metrics,
                    'timestamp': timestamp
                }
                torch.save(save_data, model_path)
                logger.info(f"✅ Modèle sauvegardé localement: {model_path}")
                
                # Vérifier que le fichier existe
                if not os.path.exists(model_path):
                    raise Exception(f"Le fichier modèle n'a pas été créé: {model_path}")
                
                # Log dans MLflow avec vérification
                try:
                    # Log du modèle
                    mlflow.pytorch.log_model(
                        pytorch_model=self.training_model,
                        artifact_path=f"models/v{self.version + 1}/{timestamp}",
                        registered_model_name="food101_continuous_learning"
                    )
                    logger.info("✅ Modèle enregistré dans MLflow")
                    
                    # Log des métriques
                    mlflow.log_metrics(metrics)
                    logger.info("✅ Métriques enregistrées dans MLflow")
                    
                    # Log des artefacts
                    mlflow.log_artifact(model_path, f"model_backups/v{self.version + 1}/{timestamp}")
                    logger.info("✅ Backup du modèle enregistré dans MLflow")
                    
                    # Log des métadonnées
                    metadata = {
                        'version': self.version + 1,
                        'timestamp': timestamp,
                        'metrics': metrics,
                        'model_file': model_filename,
                        'run_id': run.info.run_id,
                        'feedback_count': len(feedbacks)
                    }
                    metadata_path = os.path.join("/app/model_metadata", f"metadata_v{self.version + 1}_{timestamp}.json")
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=4)
                    mlflow.log_artifact(metadata_path, f"metadata/v{self.version + 1}/{timestamp}")
                    logger.info("✅ Métadonnées enregistrées dans MLflow")
                    
                except Exception as mlflow_error:
                    logger.error(f"❌ Erreur lors de l'enregistrement dans MLflow: {str(mlflow_error)}")
                    raise
                
                logger.info(f"✅ Entraînement v{self.version + 1} terminé")
                logger.info(f"📊 Métriques: {metrics}")
                logger.info(f"🔍 Run ID MLflow: {run.info.run_id}")
                logger.info(f"💾 Modèle sauvegardé: {model_filename}")
                
                return metrics
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'entraînement: {str(e)}")
            raise

    def get_prediction_model(self):
        """Retourne toujours le modèle de base pour les prédictions"""
        return self.base_model

# Variables globales
model_name = os.getenv('MODEL_NAME', "ResNet-50_acc78.61_20250221_132302.pth")
model_path = os.getenv('MODEL_PATH', f"/app/saved_models/20250221_132302/{model_name}")

def prepare_model(model):
    """Prépare le modèle avec la même architecture que lors de l'entraînement"""
    if hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 101)
        )
    return model

def load_model():
    """Charge le modèle ResNet-50"""
    try:
        global model, device
        
        # Vérifier CUDA
        logger.info("🔍 Vérification de CUDA...")
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA disponible: {cuda_available}")
        
        device = torch.device("cuda" if cuda_available else "cpu")
        if not cuda_available:
            logger.info("Mode CPU activé")
            
        # Charger le modèle de base
        model_name = "ResNet-50"
        base_model = models_to_test[model_name]["model"]
        base_model = prepare_model(base_model)  # Ajout de la préparation du modèle
        base_model = base_model.to(device)
        
        # Charger les poids pré-entraînés
        model_path = "/app/saved_models/20250221_132302/ResNet-50_acc78.61_20250221_132302.pth"
        if os.path.exists(model_path):
            logger.info(f"✅ Modèle de base trouvé: {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            
            # Extraire l'état du modèle du checkpoint
            if "model_state_dict" in checkpoint:
                base_model.load_state_dict(checkpoint["model_state_dict"])
            else:
                base_model.load_state_dict(checkpoint)
                
            logger.info("✅ Checkpoint du modèle de base chargé avec succès")
            base_model.eval()
            
            # Assigner le modèle de base à la variable globale
            model = base_model
            return True
        else:
            logger.error(f"❌ Modèle non trouvé: {model_path}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement du modèle: {str(e)}")
        return False

def process_image(image_bytes):
    """Traite l'image uploadée"""
    image = Image.open(io.BytesIO(image_bytes))
    # Appliquer la même transformation que pendant l'entraînement
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user)
):
    """Prédit la classe d'une image"""
    try:
        logger.info(f"Début de la prédiction pour le fichier: {file.filename}")
        
        # Vérifier le type de fichier
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(
                status_code=400,
                detail="Format de fichier non supporté"
            )
        
        # Lecture et traitement de l'image
        logger.info("Lecture du fichier...")
        contents = await file.read()
        
        logger.info("Traitement de l'image...")
        image_tensor = process_image(contents)
        
        # Vérifier que le modèle est chargé
        if model is None:
            logger.error("Le modèle n'est pas chargé!")
            raise HTTPException(
                status_code=500,
                detail="Erreur interne: modèle non chargé"
            )
        
        logger.info("Prédiction en cours...")
        # Utiliser le modèle de base pour les prédictions
        prediction_model = continuous_learning_manager.get_prediction_model()
        prediction_model.eval()
        
        with torch.no_grad():
            output = prediction_model(image_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
            
            # Obtenir les top 5 prédictions
            top5_prob, top5_pred = torch.topk(probabilities, 5)
            top5_results = []
            
            logger.info("Chargement des classes...")
            # Charger les classes depuis le dataset Food101
            dataset = Food101(root='./data', split='test', download=False)
            classes = dataset.classes
            
            # Préparer les résultats des top 5
            for i in range(5):
                top5_results.append({
                    "classe": classes[top5_pred[i].item()],
                    "confiance": float(top5_prob[i].item()) * 100
                })
            
            # Préparer la réponse
            response = {
                "filename": file.filename,
                "predicted_class": classes[predicted.item()],
                "confidence": float(confidence.item()) * 100,
                "top5_predictions": top5_results
            }
            
            # Ajouter la prédiction au cache
            prediction_id = f"{file.filename}_{classes[predicted.item()]}_{datetime.now().strftime('%Y%m%d_%H%M')}_{secrets.token_hex(4)}"
            predictions_cache.add(prediction_id, {
                "image_tensor": image_tensor,
                "predicted_class": classes[predicted.item()],
                "confidence": float(confidence.item()) * 100
            })
            response["prediction_id"] = prediction_id
            
            logger.info(f"Prédiction réussie: {response['predicted_class']} ({response['confidence']:.2f}%)")
            return JSONResponse(content=response)
            
    except HTTPException as he:
        logger.error(f"Erreur HTTP: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}")
        logger.exception("Traceback complet:")
        return JSONResponse(
            status_code=500,
            content={"error": f"Une erreur est survenue: {str(e)}"}
        )

@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile], current_user: str = Depends(get_current_user)):
    """Prédit la classe de plusieurs images"""
    try:
        # Vérifier si des fichiers ont été envoyés
        if not files:
            raise HTTPException(
                status_code=400,
                detail="Aucun fichier n'a été envoyé"
            )
        
        # Vérifier le nombre d'images
        if len(files) > 100:
            raise HTTPException(
                status_code=400,
                detail="Le nombre maximum d'images par requête est de 100"
            )
        
        logger.info(f"Traitement d'un lot de {len(files)} images...")
        results = []
        
        # Charger les classes une seule fois
        dataset = Food101(root='./data', split='test', download=False)
        classes = dataset.classes
        
        # Utiliser le modèle de base pour les prédictions
        prediction_model = continuous_learning_manager.get_prediction_model()
        prediction_model.eval()
        
        # Traiter chaque image
        for file in files:
            try:
                # Vérifier le type de fichier
                if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    raise ValueError("Format de fichier non supporté")
                
                # Lecture et traitement de l'image
                contents = await file.read()
                image_tensor = process_image(contents)
                
                # Prédiction
                with torch.no_grad():
                    output = prediction_model(image_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    
                    # Obtenir les top 5 prédictions
                    top5_prob, top5_pred = torch.topk(probabilities, 5)
                    top5_results = []
                    
                    # Préparer les résultats des top 5
                    for i in range(5):
                        top5_results.append({
                            "classe": classes[top5_pred[0][i].item()],
                            "confiance": float(top5_prob[0][i].item()) * 100
                        })
                    
                    # Générer un ID unique pour la prédiction
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    prediction_id = f"{file.filename}_{classes[predicted[0].item()]}_{timestamp}_{uuid.uuid4().hex[:8]}"
                    
                    # Ajouter au cache
                    predictions_cache.add(prediction_id, {
                        "predicted_class": classes[predicted[0].item()],
                        "image_tensor": image_tensor,
                        "filename": file.filename
                    })
                    
                    # Ajouter le résultat
                    results.append({
                        "filename": file.filename,
                        "prediction_id": prediction_id,
                        "predicted_class": classes[predicted[0].item()],
                        "confidence": float(confidence[0].item()) * 100,
                        "top5_predictions": top5_results,
                        "status": "success"
                    })
                    
                # Remettre le curseur de fichier au début pour la prochaine utilisation
                await file.seek(0)
                    
            except Exception as img_error:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": str(img_error)
                })
                logger.error(f"Erreur lors du traitement de {file.filename}: {str(img_error)}")
        
        # Préparer la réponse globale
        response = {
            "total_images": len(files),
            "successful_predictions": len([r for r in results if r["status"] == "success"]),
            "failed_predictions": len([r for r in results if r["status"] == "error"]),
            "results": results
        }
        
        logger.info(f"Traitement du lot terminé. {response['successful_predictions']} succès, {response['failed_predictions']} échecs")
        return JSONResponse(content=response)
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Erreur lors du traitement du lot: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Une erreur est survenue: {str(e)}"}
        )

@app.post("/predict_zip")
async def predict_zip(
    file: UploadFile = File(..., description="Fichier ZIP contenant des images", media_type="application/zip"),
    current_user: str = Depends(get_current_user)
):
    """Prédit la classe des images contenues dans un fichier zip"""
    try:
        # Vérifier que le fichier existe et est un ZIP
        if not file:
            raise HTTPException(
                status_code=400,
                detail="Aucun fichier n'a été envoyé"
            )
        
        # Vérifier l'extension et le type MIME
        filename = file.filename.lower()
        content_type = file.content_type or ""
        
        if not filename.endswith('.zip') or 'zip' not in content_type:
            raise HTTPException(
                status_code=400,
                detail="Le fichier doit être au format ZIP"
            )
            
        # Vérifier que le fichier n'est pas vide
        contents = await file.read()
        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Le fichier ZIP est vide"
            )
            
        # Créer un dossier temporaire
        with tempfile.TemporaryDirectory() as temp_dir:
            # Sauvegarder le fichier ZIP
            zip_path = os.path.join(temp_dir, filename)
            with open(zip_path, 'wb') as f:
                f.write(contents)
            
            # Extraire les images
            results = []
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Lister les fichiers images
                image_files = [f for f in zip_ref.namelist() 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if not image_files:
                    raise HTTPException(
                        status_code=400,
                        detail="Aucune image trouvée dans le ZIP"
                    )
                
                # Charger les classes une seule fois
                dataset = Food101(root='./data', split='test', download=False)
                classes = dataset.classes
                
                # Utiliser le modèle de base pour les prédictions
                prediction_model = continuous_learning_manager.get_prediction_model()
                prediction_model.eval()
                
                # Traiter chaque image
                for img_file in image_files:
                    try:
                        # Extraire et lire l'image
                        zip_ref.extract(img_file, temp_dir)
                        img_path = os.path.join(temp_dir, img_file)
                        
                        with open(img_path, 'rb') as img:
                            image_tensor = process_image(img.read())
                        
                        # Prédiction
                        with torch.no_grad():
                            output = prediction_model(image_tensor)
                            probabilities = torch.nn.functional.softmax(output[0], dim=0)
                            confidence, predicted = torch.max(probabilities, 0)
                            
                            # Top 5 prédictions
                            top5_prob, top5_pred = torch.topk(probabilities, 5)
                            top5_results = []
                            
                            for i in range(5):
                                top5_results.append({
                                    "classe": classes[top5_pred[i].item()],
                                    "confiance": float(top5_prob[i].item()) * 100
                                })
                            
                            # Générer un ID unique pour la prédiction
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            prediction_id = f"{img_file}_{classes[predicted.item()]}_{timestamp}_{uuid.uuid4().hex[:8]}"
                            
                            # Ajouter au cache
                            predictions_cache.add(prediction_id, {
                                "predicted_class": classes[predicted.item()],
                                "image_tensor": image_tensor,
                                "filename": img_file
                            })
                            
                            results.append({
                                "filename": img_file,
                                "prediction_id": prediction_id,
                                "predicted_class": classes[predicted.item()],
                                "confidence": float(confidence.item()) * 100,
                                "top5_predictions": top5_results,
                                "status": "success"
                            })
                            
                    except Exception as img_error:
                        results.append({
                            "filename": img_file,
                            "status": "error",
                            "error": str(img_error)
                        })
                        logger.error(f"Erreur lors du traitement de {img_file}: {str(img_error)}")
            
            # Préparer la réponse globale
            response = {
                "total_images": len(image_files),
                "successful_predictions": len([r for r in results if r["status"] == "success"]),
                "failed_predictions": len([r for r in results if r["status"] == "error"]),
                "results": results
            }
            
            logger.info(f"Traitement du ZIP terminé. {response['successful_predictions']} succès, {response['failed_predictions']} échecs")
            return JSONResponse(content=response)
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Erreur lors du traitement du ZIP: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Une erreur est survenue: {str(e)}"}
        )

@app.post("/auth/register")
async def register(
    username: str = Form(...),
    password: str = Form(...),
    current_user: str = Depends(get_current_user)
):
    """Crée un nouvel utilisateur (admin seulement)"""
    if api_manager.users[current_user]["role"] != "admin":
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Accès réservé aux administrateurs"
        )
    
    if api_manager.create_user(username, password):
        return {"message": f"Utilisateur {username} créé avec succès"}
    raise HTTPException(
        status_code=400,
        detail="Nom d'utilisateur déjà pris"
    )

@app.post("/auth/change-password")
async def change_password(
    old_password: str = Form(...),
    new_password: str = Form(...),
    current_user: str = Depends(get_current_user)
):
    """Change le mot de passe de l'utilisateur connecté"""
    if not api_manager.change_password(current_user, old_password, new_password):
        raise HTTPException(
            status_code=400,
            detail="Ancien mot de passe incorrect"
        )
    return {"message": "Mot de passe changé avec succès"}

@app.post("/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authentifie un utilisateur et retourne un token JWT"""
    try:
        username = form_data.username
        password = form_data.password
        
        # Vérification de l'utilisateur
        if username not in api_manager.users:
            raise HTTPException(status_code=401)
        
        if not api_manager.verify_password(username, password):
            raise HTTPException(status_code=401)
        
        # Créer le token avec les informations utilisateur
        access_token = create_access_token(
            data={
                "sub": username,
                "role": api_manager.users[username]["role"]
            }
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer"
        }
    except Exception as e:
        logger.error(f"Erreur de connexion: {str(e)}")
        raise HTTPException(status_code=401)

# Nouvel endpoint pour obtenir les informations de l'utilisateur courant
@app.get("/auth/me")
async def read_users_me(current_user: str = Depends(get_current_user)):
    """Retourne les informations de l'utilisateur connecté"""
    return {
        "username": current_user,
        "role": api_manager.users[current_user]["role"]
    }

# Endpoint pour lister tous les utilisateurs (admin seulement)
@app.get("/admin/users")
async def list_users(current_user: str = Depends(get_current_user)):
    """Liste tous les utilisateurs (admin seulement)"""
    if api_manager.users[current_user]["role"] != "admin":
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Accès réservé aux administrateurs"
        )
    return {
        "users": [
            {
                "username": username,
                "role": user["role"]
            }
            for username, user in api_manager.users.items()
        ]
    }

# Endpoint pour supprimer un utilisateur (admin seulement)
@app.delete("/admin/users/{username}")
async def delete_user(
    username: str,
    current_user: str = Depends(get_current_user)
):
    """Supprime un utilisateur (admin seulement)"""
    if api_manager.users[current_user]["role"] != "admin":
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Accès réservé aux administrateurs"
        )
    if username == current_user:
        raise HTTPException(
            status_code=400,
            detail="Impossible de supprimer son propre compte"
        )
    if username not in api_manager.users:
        raise HTTPException(
            status_code=404,
            detail="Utilisateur non trouvé"
        )
    del api_manager.users[username]
    api_manager._save_users()
    return {"message": f"Utilisateur {username} supprimé"}

async def check_and_train():
    """Vérifie si l'entraînement automatique doit être lancé"""
    if len(feedback_cache) >= FEEDBACK_THRESHOLD:
        logger.info(f"🎯 Buffer plein! ({len(feedback_cache)} feedbacks)")
        logger.info("✨ Démarrage de l'entraînement automatique...")
        try:
            await continuous_learning_manager.train_on_feedback(list(feedback_cache))
            feedback_cache.clear()  # Réinitialiser le buffer après l'entraînement
            logger.info("✅ Entraînement automatique terminé avec succès")
            logger.info("🔄 Buffer réinitialisé")
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'entraînement automatique: {str(e)}")
            # Même en cas d'erreur, on vide le buffer pour éviter les boucles
            feedback_cache.clear()
            logger.info("🔄 Buffer réinitialisé après erreur")

@app.post("/feedback")
async def feedback(
    prediction_id: str = Form(...),
    is_correct: bool = Form(...),
    correct_class: str = Form(None),
    current_user: str = Depends(get_current_user)
):
    """Soumet un feedback sur une prédiction"""
    try:
        prediction = predictions_cache.get(prediction_id)
        if not prediction:
            raise HTTPException(status_code=404, detail="Prédiction non trouvée")

        if is_correct:
            correct_class = prediction.get("predicted_class")
        elif not correct_class:
                raise HTTPException(
                    status_code=400,
                detail="correct_class est requis quand is_correct=False"
            )

        dataset = Food101(root='./data', split='test', download=False)
        if correct_class not in dataset.classes:
            raise HTTPException(
                status_code=400,
                detail=f"Classe non valide: '{correct_class}'"
            )
        
        feedback_data = {
            "prediction_id": prediction_id,
            "is_correct": is_correct,
            "used_class": correct_class,
            "timestamp": datetime.now().isoformat(),
            "image_tensor": prediction.get("image_tensor"),
            "original_class": prediction.get("predicted_class")
        }
        
        feedback_cache.append(feedback_data)
        logger.info(f"✅ Feedback ajouté: {prediction.get('predicted_class')} -> {correct_class} (correct: {is_correct})")
        
        await check_and_train()
        
        return {"status": "success", "message": "Feedback enregistré"}
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du traitement du feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_model(current_user: str = Depends(get_current_user)):
    """Lance l'entraînement du modèle sur les feedbacks accumulés"""
    try:
        if len(feedback_cache) < FEEDBACK_THRESHOLD:
            return {
                "status": "pending",
                "message": f"Pas assez de feedbacks ({len(feedback_cache)}/{FEEDBACK_THRESHOLD})"
            }

        logger.info(f"🎯 Démarrage de l'entraînement avec {len(feedback_cache)} feedbacks")
        await continuous_learning_manager.train_on_feedback(list(feedback_cache))
        feedback_cache.clear()  # Réinitialiser le buffer après l'entraînement
        logger.info("🔄 Buffer réinitialisé")
        
        return {
            "status": "success",
            "message": "Entraînement terminé avec succès"
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'entraînement: {str(e)}")
        feedback_cache.clear()  # Réinitialiser même en cas d'erreur
        logger.info("🔄 Buffer réinitialisé après erreur")
        raise HTTPException(status_code=500, detail=str(e))

def setup_mlflow():
    """Configure MLflow pour le tracking des expériences"""
    try:
        # Séparer les dossiers MLflow et modèles
        mlflow_dirs = [
            "/app/mlruns",  # Pour le tracking MLflow
            "/app/mlflow_artifacts",  # Pour les artefacts MLflow
            "/app/mlflow_artifacts/models",  # Pour les modèles MLflow
            "/app/mlflow_artifacts/metadata"  # Pour les métadonnées MLflow
        ]
        
        model_dirs = [
            "/app/saved_models",  # Pour les sauvegardes locales des modèles
            "/app/model_backups",  # Pour les backups des modèles
            "/app/model_metadata"  # Pour les métadonnées des modèles
        ]
        
        # Créer tous les dossiers nécessaires
        for directory in mlflow_dirs + model_dirs:
            os.makedirs(directory, exist_ok=True)
            os.chmod(directory, 0o777)
            logger.info(f"✅ Dossier créé/vérifié: {directory}")

        # Configuration MLflow
        mlflow_process = subprocess.Popen([
            "mlflow", "server",
            "--host", "0.0.0.0",
            "--port", "5000",
            "--backend-store-uri", "file:///app/mlruns",
            "--default-artifact-root", "file:///app/mlflow_artifacts",
            "--workers", "1"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        time.sleep(5)
        
        # Configuration de l'URI MLflow
        os.environ['MLFLOW_TRACKING_URI'] = "http://0.0.0.0:5000"
        mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
        
        # Créer ou récupérer l'expérience
        experiment_name = "food101_continuous_learning"
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                # Créer une nouvelle expérience
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location="file:///app/mlflow_artifacts"
                )
                logger.info(f"✨ Nouvelle expérience MLflow créée: {experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"📊 Expérience MLflow existante: {experiment_name}")
            
            # Définir l'expérience active
            mlflow.set_experiment(experiment_name)
            logger.info(f"✅ Expérience MLflow active: {experiment_name}")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création/récupération de l'expérience: {str(e)}")
            raise
        
        return mlflow_process
        
    except Exception as e:
        logger.error(f"❌ Erreur configuration MLflow: {str(e)}")
        return None

async def test_continuous_learning():
    """Test l'apprentissage continu au démarrage avec des images de test"""
    try:
        logger.info("🧪 Test de l'apprentissage continu au démarrage...")
        
        # Sauvegarder l'état original du modèle
        original_state = copy.deepcopy(model.state_dict())
        
        # Chemin vers le dossier des images de test
        test_images_dir = "/app/image-test-apprentissage"
        if not os.path.exists(test_images_dir):
            logger.error(f"❌ Dossier {test_images_dir} non trouvé")
            return
            
        # Vérifier le contenu du dossier des images
        logger.info("📂 Contenu du dossier image-test-apprentissage:")
        logger.info(os.listdir(test_images_dir))
        
        # Liste des images de test (limiter à 10)
        image_files = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))][:10]
        if not image_files:
            logger.error("❌ Aucune image trouvée dans le dossier de test")
            return
            
        logger.info(f"📸 {len(image_files)} images trouvées pour le test: {image_files}")
        
        # Prédiction et feedback pour chaque image
        predictions = []
        for image_file in image_files:
            image_path = os.path.join(test_images_dir, image_file)
            with open(image_path, 'rb') as f:
                image_content = f.read()
                
            file = UploadFile(
                file=io.BytesIO(image_content),
                filename=image_file,
                headers={"content-type": "image/jpeg"}
            )
            
            response = await predict(file, "system_test")
            prediction_data = response.body.decode()
            prediction = json.loads(prediction_data)
            predictions.append(prediction)
            logger.info(f"✅ Prédiction faite pour {image_file}")
            
        # Envoyer les feedbacks
        logger.info("🎯 Envoi des feedbacks...")
        for pred in predictions:
            try:
                await feedback(
                    prediction_id=pred["prediction_id"],
                    is_correct=True,
                    correct_class=pred["predicted_class"],
                    current_user="system_test"
                )
                logger.info(f"✅ Feedback envoyé pour {pred['filename']}")
            except Exception as e:
                logger.error(f"❌ Erreur lors de l'envoi du feedback pour {pred['filename']}: {str(e)}")
        
        # Restaurer l'état original
        logger.info("🔄 Restauration du modèle original...")
        model.load_state_dict(original_state)
        model.eval()
        logger.info("✅ Modèle original restauré")
        
        logger.info("✨ Test d'apprentissage continu terminé")
        
    except Exception as e:
        if original_state:
            model.load_state_dict(original_state)
            model.eval()
            logger.info("🔄 Modèle original restauré après erreur")
        logger.error(f"❌ Erreur lors du test d'apprentissage continu: {str(e)}")

def create_system_user():
    """Crée un utilisateur système pour les tests"""
    system_user = {
        "username": "system_test",
        "password": secrets.token_hex(16),
        "role": "system",
        "api_keys": {}
    }
    if "system_test" not in api_manager.users:
        api_manager.users["system_test"] = system_user
    return system_user

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 