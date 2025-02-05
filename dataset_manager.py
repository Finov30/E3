import os
from torchvision.datasets import Food101
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class DatasetManager:
    def __init__(self, transform, data_dir="./data"):
        self.transform = transform
        self.data_dir = data_dir
        
    def check_and_prepare_dataset(self):
        """Vérifie si le dataset existe et le télécharge si nécessaire"""
        print("\nVérification du dataset Food-101...")
        
        # Vérifie si le dossier data existe
        if not os.path.exists(self.data_dir):
            print("Dossier data non trouvé. Création du dossier...")
            os.makedirs(self.data_dir)
        
        # Vérifie si le dataset est déjà téléchargé
        dataset_path = os.path.join(self.data_dir, 'food-101')
        if not os.path.exists(dataset_path):
            print("Dataset non trouvé. Téléchargement en cours...")
            self._download_dataset()
        else:
            print("Dataset trouvé!")
            
    def get_dataloaders(self, batch_size, num_samples=None):
        """Retourne les dataloaders pour l'entraînement et le test"""
        # Chargement des datasets
        dataset_train = Food101(root=self.data_dir, split="train", 
                              transform=self.transform, download=False)
        dataset_test = Food101(root=self.data_dir, split="test", 
                             transform=self.transform, download=False)
        
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
    
    def _download_dataset(self):
        """Télécharge le dataset"""
        try:
            # Le download=True va télécharger le dataset
            Food101(root=self.data_dir, split="train", download=True)
            print("Dataset téléchargé avec succès!")
        except Exception as e:
            print(f"Erreur lors du téléchargement du dataset: {str(e)}")
            raise 