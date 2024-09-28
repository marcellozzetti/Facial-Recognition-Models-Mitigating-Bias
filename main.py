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
NUM_EPOCHS = 10
LEARNING_RATES = [0.01, 0.001, 0.0001]
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Check if Cuda is available
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

if torch.cuda.is_available() and device == torch.device("cuda"):
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

# Transformations and normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

csv_pd = pd.read_csv(pre_processing_images.CSV_BALANCED_CONCAT_DATASET_FILE)
dataset = FaceDataset(csv_pd, pre_processing_images.IMG_BASE_DIR, transform=transform)

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

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.ls_eps = ls_eps
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine - self.m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > (1.0 - self.m), phi, cosine)
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            phi = phi - self.ls_eps
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        return output

# Define the model (LResNet50E-IR, a modified ResNet50 for ArcFace)
class LResNet50E_IR(nn.Module):
    def __init__(self, num_classes=len(label_encoder.classes_)):
        super(LResNet50E_IR, self).__init__()
        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.backbone.fc = self.fc

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        return x

def adjust_learning_rate(optimizer, epoch):
    if epoch < 10:
        lr = LEARNING_RATES[0]
    elif epoch < 15:
        lr = LEARNING_RATES[1]
    else:
        lr = LEARNING_RATES[2]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Initialize model, criterion, and optimizer
model = LResNet50E_IR().to(device)
model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1, verbose=True)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

print("Step 9 (CNN model): End")


print("Step 10 (Training execution): Start")

# Training execution
num_epochs = NUM_EPOCHS
train_losses = []
accuracies = []
precisions = []
log_losses = []

scaler =  torch.amp.GradScaler(torch.device(device)) if device == torch.device("cuda") else None

for epoch in range(num_epochs):
    #adjust_learning_rate(optimizer, epoch)
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

    overhead = time.time() - start_time
    print(f'Epoch {epoch+1}/{num_epochs}, Overhead: {overhead:.4f}s')

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

            all_labels.extend(labels_tensor)
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

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}, '
          f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Log Loss: {logloss:.4f}')

    del images, labels, outputs, loss
    gc.collect()
    if torch.cuda.is_available() and device == torch.device("cuda"):
        torch.cuda.empty_cache()

torch.save(model.state_dict(), pre_processing_images.MODEL_FAIRFACE_FILE)
print('Finished Training and Model Saved')

# Plotting general metrics
epochs_range = range(1, num_epochs + 1)
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


