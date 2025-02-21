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
        
    def dataset_exists(self):
        """Vérifie si le dataset Food-101 existe déjà"""
        data_path = Path('./data/food-101')
        # Vérifie l'existence des fichiers essentiels
        required_files = [
            data_path / 'meta' / 'classes.txt',
            data_path / 'license_agreement.txt'
        ]
        return all(f.exists() for f in required_files)

    def check_and_prepare_dataset(self):
        """Vérifie et prépare le dataset si nécessaire"""
        if self.dataset_exists():
            print("✓ Dataset Food-101 déjà présent")
            return
        
        print("Téléchargement et préparation du dataset Food-101...")
        # Télécharge le dataset si nécessaire
        self.train_dataset = Food101(root='./data', split='train', 
                                    download=True, transform=self.transform)
        self.test_dataset = Food101(root='./data', split='test', 
                                   download=True, transform=self.transform)
        
    def get_dataloaders(self, batch_size=64, num_samples=None):
        """Retourne les dataloaders pour l'entraînement et le test"""
        try:
            print("Chargement du dataset...")
            
            # Chargement des datasets
            dataset_train = Food101(root='./data', split='train', 
                                  transform=self.transform, download=False)
            dataset_test = Food101(root='./data', split='test', 
                                 transform=self.transform, download=False)
            
            # Si num_samples est spécifié, on réduit la taille du dataset
            if num_samples is not None:
                # Création de sous-ensembles pour le test
                indices_train = torch.randperm(len(dataset_train))[:num_samples]
                indices_test = torch.randperm(len(dataset_test))[:num_samples]
                dataset_train = torch.utils.data.Subset(dataset_train, indices_train)
                dataset_test = torch.utils.data.Subset(dataset_test, indices_test)
            
            train_loader = DataLoader(
                dataset_train, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=2,  # Réduit de 4 à 2 pour les tests
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            test_loader = DataLoader(
                dataset_test, 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=2,  # Réduit de 4 à 2 pour les tests
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            return train_loader, test_loader
            
        except Exception as e:
            print(f"Erreur lors du chargement du dataset: {str(e)}")
            if not self.dataset_exists():
                print("Tentative de téléchargement...")
                self.check_and_prepare_dataset()
                # Retry after download
                return self.get_dataloaders(batch_size, num_samples)
            else:
                raise
    
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