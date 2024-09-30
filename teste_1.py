print("Step 1 (Imports): Start")

import os
import time
import json
import math
import ssl
import gc
import io
import zipfile
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle
from PIL import Image
from scipy.spatial import distance
from sklearn.metrics import accuracy_score, precision_score, log_loss
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import cv2
from mtcnn.mtcnn import MTCNN
import pre_processing_images

# Constants
BATCH_SIZE = 256
NUM_EPOCHS = 40
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
learning_rate = 0.001

# Check if Cuda is available
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

if torch.cuda.is_available() and device == torch.device("cuda"):
        gc.collect()
        torch.cuda.empty_cache()

print("Step 1 (Imports): End")


print("Step 9 (CNN model): Start")

# Custom Dataset using CSV for labels
class FaceDataset(Dataset):
    def __init__(self, csv_pd, img_dir, transform=None):
        self.labels_df = csv_pd
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.labels_df.iloc[idx, 0])
        label = self.labels_df.iloc[idx, 3]  # 'race' is the class

        try:
            img = cv2.imread(img_name)
            if img is None:
                raise FileNotFoundError(f"Image {img_name} not found")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = Image.fromarray(img)
                img = self.transform(img)
            return img, label
        except (FileNotFoundError, ValueError) as e:
            return None, None

transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

csv_pd = pd.read_csv(pre_processing_images.CSV_BALANCED_CONCAT_DATASET_FILE)
dataset = FaceDataset(csv_pd, pre_processing_images.IMG_PROCESSED_DIR, transform=transform)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

def collate_fn(batch):
    batch = [item for item in batch if item[0] is not None and item[1] is not None]
    if len(batch) == 0:
        return torch.empty(0), torch.empty(0)
    return torch.utils.data.dataloader.default_collate(batch)

label_encoder = LabelEncoder()
label_encoder.fit(csv_pd['race'])

# Create DataLoaders using a filter function: collate_fn
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True, collate_fn=collate_fn)

# Definindo o modelo ResNet50 pré-treinado
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features  # Isso deve retornar 2048
model.fc = nn.Identity()  # Mantém a saída de 2048
model = model.to(device)

# Definindo a camada ArcFace
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcMarginProduct, self).__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))  # [n_classes, in_features]
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # Normalizando entrada e pesos
        cosine = nn.functional.linear(nn.functional.normalize(input), nn.functional.normalize(self.weight))  # MatMul
        theta = torch.acos(cosine.clamp(-1.0, 1.0))  # Theta = arccos(cosine)
        target_logit = torch.cos(theta + self.m)  # Aplica margem

        # One-hot encoding dos rótulos
        one_hot = torch.zeros_like(cosine)  # Cria um tensor vazio de mesma forma que cosine
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)  # Marca as classes corretas

        # Combina logits com margem e ajusta escala
        output = (one_hot * target_logit) + ((1.0 - one_hot) * cosine)
        output *= self.s  # Multiplica pela escala

        return output

arcface = ArcMarginProduct(num_ftrs, len(csv_pd['race'].unique())).to(device)

label_encoder = LabelEncoder()
label_encoder.fit(csv_pd['race'])

# Definindo o otimizador e a função de perda
optimizer = optim.AdamW([{'params': model.parameters()}, {'params': arcface.parameters()}], lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Definindo o scheduler para ajustar a taxa de aprendizado
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# GradScaler para Mixed Precision
scaler = torch.amp.GradScaler() if device == torch.device("cuda") else None

print("Step 9 (CNN model): End")


print("Step 10 (Training execution): Start")

# Training execution
train_losses = []
accuracies = []
precisions = []
log_losses = []

for epoch in range(NUM_EPOCHS):
    model.train()
    start_time = time.time()
    
    for images, labels in train_loader:
        images = images.to(device)
        labels_tensor = torch.tensor(label_encoder.transform(labels)).to(device)
        optimizer.zero_grad()

        if device == torch.device("cuda"):
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels_tensor)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels_tensor)

        if scaler and device == torch.device("cuda"):
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        #scheduler.step()

    overhead = time.time() - start_time
        
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Overhead: {overhead:.4f}s')
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Learning rate: {scheduler.get_last_lr()}")

    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    epoch_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels_tensor = torch.tensor(label_encoder.transform(labels)).to(device)
            outputs = model(images)
            loss = criterion(outputs, labels_tensor)
            epoch_loss += loss.item()

            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = torch.max(outputs, 1)[1].cpu().numpy()

            all_labels.extend(labels_tensor.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    all_labels = [label.item() for label in all_labels]
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    all_probs = softmax(all_probs)
    logloss = log_loss(all_labels, all_probs)

    scheduler.step(epoch_loss)

    train_losses.append(epoch_loss / len(val_loader))
    accuracies.append(accuracy)
    precisions.append(precision)
    log_losses.append(logloss)

    print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss/len(train_loader):.4f}, '
          f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Log Loss: {logloss:.4f}')

    del images, labels, outputs, loss
    gc.collect()
    if torch.cuda.is_available() and device == torch.device("cuda"):
        torch.cuda.empty_cache()

torch.save(model.state_dict(), pre_processing_images.MODEL_FAIRFACE_FILE)
print('Finished Training and Model Saved')

# Plotting general metrics
epochs_range = range(1, NUM_EPOCHS + 1)
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(epochs_range, train_losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(epochs_range, accuracies, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(epochs_range, precisions, label='Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Precision over Epochs')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(epochs_range, log_losses, label='Log Loss')
plt.xlabel('Epoch')
plt.ylabel('Log Loss')
plt.title('Log Loss over Epochs')
plt.legend()

plt.tight_layout()
plt.savefig('output/training_metrics.png')
plt.show()
plt.close()

print("Step 11 (Training execution): End")

print("Step 12 (Testing): Start")
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True, collate_fn=collate_fn)

model.eval()
all_test_preds = []
all_test_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels_tensor = torch.tensor(label_encoder.transform(labels)).to(device)
        outputs = model(images)
        preds = torch.max(outputs, 1)[1].cpu().numpy()

        all_test_labels.extend(labels_tensor.cpu().numpy())
        all_test_preds.extend(preds)

test_accuracy = accuracy_score(all_test_labels, all_test_preds)
print(f'Test Accuracy: {test_accuracy:.4f}')

print("Step 12 (Testing): End")
