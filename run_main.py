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
from tqdm import tqdm
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
from face_dataset import FaceDataset, dataset_transformation
from models import LResNet50E_IR
import pre_processing_images

# Hiperparameters
BATCH_SIZE = 4
NUM_EPOCHS = 24
TRAIN_VAL_SPLIT = 0.8
VAL_VAL_SPLIT = 0.1
LEARNING_RATE = 0.001
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Check if Cuda is available
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

#Clean memory
if torch.cuda.is_available() and device == torch.device("cuda"):
        gc.collect()
        torch.cuda.empty_cache()


print("Step 9 (CNN model): Start")

csv_pd = pd.read_csv(pre_processing_images.CSV_BALANCED_CONCAT_DATASET_FILE)
dataset = FaceDataset(csv_pd, pre_processing_images.IMG_PROCESSED_DIR, transform=dataset_transformation)

# Split dataset into training and validation sets
train_size = int(TRAIN_VAL_SPLIT * len(dataset))
val_size = int(VAL_VAL_SPLIT * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

label_encoder = LabelEncoder()
label_encoder.fit(csv_pd['race'])
num_classes = len(dataset.classes)

# Create DataLoaders using a filter function: collate_fn
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)

# Initialize model, criterion, and optimizer
model = LResNet50E_IR(num_classes).to(device)
model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
#scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, epochs=NUM_EPOCHS, steps_per_epoch=len(train_loader))
scaler =  torch.amp.GradScaler(torch.device(device)) if device == torch.device("cuda") else None

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

print("Step 9 (CNN model): End")


print("Step 10 (Training execution): Start")
# Inicializando listas para armazenar as métricas
train_losses, val_losses, accuracies, precisions, log_losses = [], [], [], [], []

# Função de treino
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        start_time = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Modo de treinamento
            else:
                model.eval()   # Modo de validação

            running_loss = 0.0
            running_corrects = 0
            all_probs = []
            all_labels = []

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels_tensor = torch.tensor(label_encoder.transform(labels)).to(device)

                optimizer.zero_grad()

                with torch.amp.autocast("cuda", enabled=device == torch.device("cuda")), \
                     torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                         
                    probs = F.softmax(outputs, dim=1)
                    _, preds = torch.max(probs, 1)

                    loss = criterion(outputs, labels_tensor)  # Usar as saídas para calcular a perda

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                # Acumule métricas
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels_tensor.data)

                # Coleta de rótulos e probabilidades
                all_labels.extend(labels_tensor.cpu().numpy())
                all_probs.extend(probs.detach().cpu().numpy())

            # Cálculo do log_loss
            try:
                log_loss_value = log_loss(all_labels, all_probs)
                log_losses.append(log_loss_value)
            except ValueError as e:
                print(f"Erro ao calcular log_loss: {e}")

            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_loss = running_loss / dataset_sizes[phase]

            elapsed_time = time.time() - start_time
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Log Loss: {log_losses[-1]:.4f} Overhead: {elapsed_time:.2f}s')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        if phase == 'train':
            scheduler.step()  # Ajusta a taxa de aprendizado apenas no treinamento

    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)

    del images, labels, outputs, loss
    gc.collect()
    if torch.cuda.is_available() and device == torch.device("cuda"):
        torch.cuda.empty_cache()
            
    return model

# Treinando o modelo
model = train_model(model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS)

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
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)

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
