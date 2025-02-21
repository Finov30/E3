import torch
from bench_config import models_to_test, device, transform
from bench_utils import load_trained_model
from dataset_manager import DatasetManager

def load_and_use_model():
    # 1. Créer une instance vide du modèle
    model = models_to_test["ResNet-50"]
    
    # 2. Charger les poids sauvegardés
    # Remplacez le chemin par celui de votre modèle sauvegardé
    model_path = "saved_models/ResNet-50_20240219_144002.pth"
    loaded_model, saved_metrics = load_trained_model(model, model_path)
    
    if loaded_model is None:
        print("Erreur lors du chargement du modèle")
        return
    
    # 3. Mettre le modèle en mode évaluation
    loaded_model.eval()
    loaded_model.to(device)
    
    print("\nMétriques du modèle chargé:")
    for metric, value in saved_metrics.items():
        print(f"{metric}: {value}")
    
    # 4. Vous pouvez maintenant utiliser le modèle pour faire des prédictions
    # Par exemple, charger une image de test :
    dataset_manager = DatasetManager(transform)
    _, test_loader = dataset_manager.get_dataloaders(batch_size=1)
    
    # Faire une prédiction sur une image
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = loaded_model(images)
            _, predicted = torch.max(outputs, 1)
            print(f"\nPrédiction: Classe {predicted.item()}")
            break  # On ne fait qu'une seule prédiction pour l'exemple

if __name__ == "__main__":
    load_and_use_model() 