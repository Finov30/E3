import os
from torchvision.datasets import Food101
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import torch

class DatasetManager:
    def __init__(self, transform, data_dir="data"):
        self.transform = transform
        self.root_dir = Path(__file__).parent.parent
        self.data_dir = self.root_dir / data_dir
        
    def check_and_prepare_dataset(self):
        """Vérifie si le dataset existe et le télécharge si nécessaire"""
        print("\nVérification du dataset Food-101...")
        
        # Vérifie si le dossier data existe
        if not self.data_dir.exists():
            print("Dossier data non trouvé. Création du dossier...")
            self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Vérifie si le dataset est déjà téléchargé
        dataset_path = self.data_dir / 'food-101'
        meta_path = dataset_path / 'meta'
        images_path = dataset_path / 'images'
        
        if not all(p.exists() for p in [dataset_path, meta_path, images_path]):
            print("Dataset incomplet ou non trouvé. Téléchargement en cours...")
            self._download_dataset()
        else:
            print("Dataset trouvé!")
            
    def get_dataloaders(self, batch_size=32, num_samples=None):
        """Retourne les dataloaders pour l'entraînement et le test"""
        try:
            # Chargement des datasets avec download=True au cas où
            dataset_train = Food101(root=str(self.data_dir), split="train", 
                                  transform=self.transform, download=True)
            dataset_test = Food101(root=str(self.data_dir), split="test", 
                                 transform=self.transform, download=True)
            
            if num_samples:
                # Assurer un minimum d'images par classe
                min_samples_per_class = 20
                num_samples = max(num_samples, min_samples_per_class * 101)
                
                # Stratifier l'échantillonnage pour garder toutes les classes
                indices = torch.randperm(len(dataset_train))[:num_samples]
                dataset_train = torch.utils.data.Subset(dataset_train, indices)
                
                # Garder un ratio test/train cohérent
                test_size = int(num_samples * 0.2)
                test_indices = torch.randperm(len(dataset_test))[:test_size]
                dataset_test = torch.utils.data.Subset(dataset_test, test_indices)
            
            # Création des dataloaders
            train_loader = DataLoader(
                dataset_train, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=4,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            test_loader = DataLoader(
                dataset_test, 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=4,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            return train_loader, test_loader
            
        except Exception as e:
            print(f"Erreur lors du chargement du dataset: {str(e)}")
            print("Tentative de téléchargement...")
            self._download_dataset()
            # Retry after download
            return self.get_dataloaders(batch_size, num_samples)
    
    def _download_dataset(self):
        """Télécharge le dataset"""
        try:
            print("Téléchargement du dataset Food-101...")
            # Force le téléchargement
            Food101(root=str(self.data_dir), split="train", download=True)
            Food101(root=str(self.data_dir), split="test", download=True)
            print("Dataset téléchargé avec succès!")
        except Exception as e:
            print(f"Erreur lors du téléchargement du dataset: {str(e)}")
            raise