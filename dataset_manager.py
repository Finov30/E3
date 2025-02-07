import os
from torchvision.datasets import Food101
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pathlib import Path

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
            
    def get_dataloaders(self, batch_size, num_samples=None):
        """Retourne les dataloaders pour l'entraînement et le test"""
        try:
            # Chargement des datasets avec download=True au cas où
            dataset_train = Food101(root=str(self.data_dir), split="train", 
                                  transform=self.transform, download=True)
            dataset_test = Food101(root=str(self.data_dir), split="test", 
                                 transform=self.transform, download=True)
            
            # Si num_samples est spécifié, on réduit la taille du dataset
            if num_samples is not None:
                from torch.utils.data import Subset
                import torch
                dataset_train = Subset(dataset_train, range(num_samples))
                dataset_test = Subset(dataset_test, range(num_samples//2))
            
            # Création des dataloaders
            train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
            
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