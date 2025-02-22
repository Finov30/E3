import torch
import random
from pathlib import Path
import json
from torchvision.datasets import Food101
from torch.utils.data import DataLoader, Subset
from bench_config import transform, device, models_to_test
import os
from bench_utils import load_saved_model
import torch.nn as nn
from datetime import datetime

def prepare_model(model):
    """Prépare le modèle avec la même architecture que lors de l'entraînement"""
    if hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 101)
        )
    return model

def evaluate_random_images(model_path, num_images=100):
    """
    Évalue le modèle sur un ensemble aléatoire d'images
    """
    print(f"Évaluation de {num_images} images aléatoires...")
    
    try:
        # Chargement du modèle
        checkpoint = torch.load(model_path)
        model = models_to_test["ResNet-50"]  # Créer une nouvelle instance
        model = prepare_model(model)  # Préparer l'architecture
        model.load_state_dict(checkpoint['model_state_dict'])  # Charger les poids
        
        if model is None:
            raise Exception("Erreur lors du chargement du modèle")
        
        model = model.to(device)
        model.eval()
        
        # Chargement du dataset
        dataset = Food101(root='./data', split='test', transform=transform, download=False)
        
        # Obtenir les chemins des images et les classes
        image_paths = []
        for idx in range(len(dataset)):
            img_path, _ = dataset._image_files[idx], dataset._labels[idx]
            image_paths.append(img_path)
        
        # Sélection aléatoire des indices
        total_images = len(dataset)
        random_indices = random.sample(range(total_images), num_images)
        
        # Création d'un sous-ensemble avec les images sélectionnées
        subset = Subset(dataset, random_indices)
        loader = DataLoader(subset, batch_size=1, shuffle=False)
        
        results = []
        correct = 0
        
        print("\nAnalyse des images en cours...")
        
        with torch.no_grad():
            for i, (image, label) in enumerate(loader, 1):
                image = image.to(device)
                label = label.to(device)
                
                # Prédiction
                output = model(image)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # Vérification si la prédiction est correcte
                is_correct = (predicted == label).item()
                if is_correct:
                    correct += 1
                
                # Récupération du nom de l'image et de la classe
                image_path = image_paths[random_indices[i-1]]
                true_class = dataset.classes[label.item()]
                predicted_class = dataset.classes[predicted.item()]
                
                results.append({
                    'image_name': Path(image_path).name,
                    'true_class': true_class,
                    'predicted_class': predicted_class,
                    'confidence': float(confidence.item()) * 100,  # Conversion en pourcentage
                    'correct': is_correct
                })
                
                # Affichage de la progression
                if i % 10 == 0:
                    print(f"Traitement : {i}/{num_images} images")
        
        # Calcul de l'accuracy finale
        accuracy = (correct / num_images) * 100
        
        # Affichage des résultats
        print("\n" + "="*50)
        print(f"Résultats de l'évaluation sur {num_images} images:")
        print(f"Accuracy: {accuracy:.2f}%")
        print("="*50 + "\n")
        
        print("Détails des prédictions:")
        print("-"*50)
        for result in results:
            status = "✓" if result['correct'] else "✗"
            print(f"{status} Image: {result['image_name']}")
            print(f"   Classe réelle: {result['true_class']}")
            print(f"   Prédiction: {result['predicted_class']} (confiance: {result['confidence']:.2f}%)")
            print("-"*50)
        
        # Sauvegarde des résultats
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = Path("evaluation_results")
        results_path.mkdir(exist_ok=True)
        
        results_file = results_path / f"random_evaluation_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'accuracy': accuracy,
                'num_images': num_images,
                'results': results
            }, f, indent=4)
        
        print(f"\nRésultats sauvegardés dans: {results_file}")
        
    except Exception as e:
        print(f"Une erreur est survenue: {str(e)}")
        raise

if __name__ == "__main__":
    # Chemin direct vers le modèle
    model_path = "saved_models/20250221_132302/ResNet-50_acc78.61_20250221_132302.pth"
    
    if not Path(model_path).exists():
        raise Exception(f"Modèle non trouvé: {model_path}")
    
    print(f"Utilisation du modèle: {model_path}")
    evaluate_random_images(model_path) 