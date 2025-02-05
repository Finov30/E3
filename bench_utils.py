import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm

def train_model(model, train_loader, device, epochs=1):
    model.to(device)
    
    # Modification de la dernière couche en fonction du type de modèle
    if hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, 101)  # Pour ResNet
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, 101)  # Pour EfficientNet
        else:
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, 101)  # Pour MobileNetV3
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    start_time = time.time()
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        epoch_loss = running_loss/len(train_loader)
        print(f"\nEpoch {epoch+1}, Loss: {epoch_loss:.4f}")
    
    training_time = time.time() - start_time
    return training_time

def evaluate_model(model, test_loader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            pbar.set_postfix({'accuracy': f'{accuracy:.2f}%'})
    
    return 100 * correct / total 