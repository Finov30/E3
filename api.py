from fastapi import FastAPI, UploadFile, File, HTTPException, Security, Depends, Form
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

# Chargement des variables d'environnement
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application"""
    # Startup
    logger.info("Démarrage de l'API Food101...")
    load_model()
    logger.info("API prête à recevoir des requêtes!")
    yield
    # Shutdown
    logger.info("Arrêt de l'API...")
    global model
    if model is not None:
        model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

# Variables globales pour le modèle
model = None
model_name = os.getenv('MODEL_NAME', "ResNet-50_acc78.61_20250221_132302.pth")
model_path = os.getenv('MODEL_PATH', f"/app/models/20250221_132302/{model_name}")

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
    """Charge le modèle une seule fois au démarrage"""
    global model
    try:
        if model is None:
            logger.info(f"Chargement du modèle: {model_name}")
            logger.info(f"Chemin du modèle: {model_path}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
            
            # Vérifier la taille du fichier
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # Taille en MB
            logger.info(f"Taille du modèle: {file_size:.2f} MB")
            
            try:
                checkpoint = torch.load(model_path, map_location=device)
                logger.info("Checkpoint chargé avec succès")
                
                model = models_to_test["ResNet-50"]
                logger.info("Modèle ResNet-50 créé")
                
                model = prepare_model(model)
                logger.info("Modèle préparé")
                
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("État du modèle chargé")
                
                model = model.to(device)
                model.eval()
                logger.info("Modèle mis en mode évaluation")
                
            except Exception as e:
                logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
                logger.exception("Traceback complet:")
                raise
            
            logger.info("✅ Modèle chargé avec succès!")
            
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement du modèle: {str(e)}")
        logger.exception("Traceback complet:")
        raise

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
        # Prédiction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
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
                    "classe": classes[top5_pred[0][i].item()],
                    "confiance": float(top5_prob[0][i].item()) * 100
                })
            
            # Préparer la réponse
            response = {
                "filename": file.filename,
                "predicted_class": classes[predicted.item()],
                "confidence": float(confidence.item()) * 100,
                "top5_predictions": top5_results
            }
            
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
async def predict_batch(
    files: List[UploadFile] = File(...),
    current_user: str = Depends(get_current_user)
):
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
                    output = model(image_tensor)
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
                    
                    # Ajouter les résultats pour cette image
                    results.append({
                        "filename": file.filename,
                        "predicted_class": classes[predicted.item()],
                        "confidence": float(confidence.item()) * 100,
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
    zip_file: UploadFile = File(...),
    current_user: str = Depends(get_current_user)
):
    """Prédit la classe des images dans un fichier ZIP"""
    try:
        # Vérifier que c'est bien un fichier ZIP
        if not zip_file.filename.endswith('.zip'):
            raise HTTPException(
                status_code=400,
                detail="Le fichier doit être au format ZIP"
            )
        
        # Créer un dossier temporaire pour extraire les images
        with tempfile.TemporaryDirectory() as temp_dir:
            # Lire et sauvegarder le fichier ZIP
            zip_path = os.path.join(temp_dir, "images.zip")
            contents = await zip_file.read()
            with open(zip_path, 'wb') as f:
                f.write(contents)
            
            # Extraire les images
            image_files = []
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Vérifier le nombre d'images avant extraction
                image_count = sum(1 for name in zip_ref.namelist() 
                                if name.lower().endswith(('.png', '.jpg', '.jpeg')))
                
                if image_count > 100:
                    raise HTTPException(
                        status_code=400,
                        detail="Le ZIP ne doit pas contenir plus de 100 images"
                    )
                
                # Extraire les fichiers
                zip_ref.extractall(temp_dir)
                # Lister les images
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            image_files.append(os.path.join(root, file))
            
            logger.info(f"Traitement d'un lot de {len(image_files)} images depuis le ZIP...")
            results = []
            
            # Charger les classes une seule fois
            dataset = Food101(root='./data', split='test', download=False)
            classes = dataset.classes
            
            # Traiter chaque image
            for img_path in image_files:
                try:
                    # Lecture et traitement de l'image
                    with open(img_path, 'rb') as img_file:
                        image_tensor = process_image(img_file.read())
                    
                    # Prédiction
                    with torch.no_grad():
                        output = model(image_tensor)
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
                        
                        # Ajouter les résultats pour cette image
                        results.append({
                            "filename": os.path.basename(img_path),
                            "predicted_class": classes[predicted.item()],
                            "confidence": float(confidence.item()) * 100,
                            "top5_predictions": top5_results,
                            "status": "success"
                        })
                        
                except Exception as img_error:
                    results.append({
                        "filename": os.path.basename(img_path),
                        "status": "error",
                        "error": str(img_error)
                    })
                    logger.error(f"Erreur lors du traitement de {os.path.basename(img_path)}: {str(img_error)}")
            
            # Préparer la réponse globale
            response = {
                "zip_filename": zip_file.filename,
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

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 