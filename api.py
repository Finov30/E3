from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from bench_config import transform, device, models_to_test
from PIL import Image
import io
import uvicorn
from pathlib import Path
import logging

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
            from torchvision.datasets import Food101
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

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 