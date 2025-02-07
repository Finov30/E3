import torch
import torchvision.transforms as transforms
from torchvision import models

# Configuration commune
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prétraitement des images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Modèles à tester
models_to_test = {
    "ResNet-50": models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
    "EfficientNet-B0": models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "MobileNetV3": models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
}

# Configuration pour benchmark complet
FULL_CONFIG = {
    "batch_size": 32,
    "num_samples": None,  # Utilise tout le dataset
    "epochs": 1
}

# Configuration pour test rapide
QUICK_CONFIG = {
    "batch_size": 16,
    "num_samples": 100,
    "epochs": 1
} 