from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from bench_config import transform, device, models_to_test
from PIL import Image
import io
import uvicorn
from pathlib import Path
import logging
from typing import List
import zipfile
import tempfile
import os
from torchvision.datasets import Food101

app = FastAPI(
    title="Food101 Classification API",
    description="API pour classifier des images de nourriture parmi 101 classes",
    version="1.0.0"
)

# Variables globales pour le modèle
model = None
model_path = "saved_models/20250221_132302/ResNet-50_acc78.61_20250221_132302.pth"

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    if model is None:
        print("Chargement du modèle...")
        checkpoint = torch.load(model_path)
        model = models_to_test["ResNet-50"]
        model = prepare_model(model)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        print("Modèle chargé avec succès!")

@app.on_event("startup")
async def startup_event():
    """Événement exécuté au démarrage de l'API"""
    logger.info("Démarrage de l'API Food101...")
    load_model()
    logger.info("API prête à recevoir des requêtes!")

@app.on_event("shutdown")
async def shutdown_event():
    """Événement exécuté à l'arrêt de l'API"""
    logger.info("Arrêt de l'API...")

def process_image(image_bytes):
    """Traite l'image uploadée"""
    image = Image.open(io.BytesIO(image_bytes))
    # Appliquer la même transformation que pendant l'entraînement
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint pour prédire la classe d'une image
    """
    try:
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
            
            return JSONResponse(content=response)
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Une erreur est survenue: {str(e)}"}
        )

@app.post("/predict_batch/")
async def predict_batch(files: List[UploadFile] = File(None)):
    """
    Endpoint pour prédire la classe de plusieurs images (max 100)
    """
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

@app.post("/predict_zip/")
async def predict_zip(zip_file: UploadFile = File(...)):
    """
    Endpoint pour prédire la classe de plusieurs images contenues dans un fichier ZIP
    """
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

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 