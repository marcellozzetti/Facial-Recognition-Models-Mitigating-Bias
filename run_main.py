import os
import time
import json
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from sklearn.metrics import accuracy_score, precision_score, log_loss
from sklearn.preprocessing import LabelEncoder
from face_dataset import FaceDataset, dataset_transformation
from models import LResNet50E_IR
import pre_processing_images

# Hyperparameters
BATCH_SIZE = 4
NUM_EPOCHS = 24
TRAIN_VAL_SPLIT = 0.8
VAL_VAL_SPLIT = 0.1
LEARNING_RATE = 0.001

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clean memory
def clean_memory():
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

clean_memory()

print("Step 9 (CNN model): Start")

# Load dataset
csv_pd = pd.read_csv(pre_processing_images.CSV_BALANCED_CONCAT_DATASET_FILE)
dataset = FaceDataset(csv_pd, pre_processing_images.IMG_PROCESSED_DIR, transform=dataset_transformation)

# Split dataset into training, validation, and test sets
train_size = int(TRAIN_VAL_SPLIT * len(dataset))
val_size = int(VAL_VAL_SPLIT * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)

label_encoder = LabelEncoder()
label_encoder.fit(csv_pd['race'])
num_classes = len(dataset.classes)

# Initialize model, criterion, and optimizer
model = LResNet50E_IR(num_classes).to(device)
model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)
scaler =  torch.amp.GradScaler(torch.device(device)) if device == torch.device("cuda") else None

print("Step 9 (CNN model): End")

print("Step 10 (Training execution): Start")
# Initializing metrics lists
train_losses, val_losses, accuracies, precisions, log_losses = [], [], [], [], []

# Training function
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    
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

    clean_memory()
    
    return model

# Train the model
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
model.eval()
all_test_preds = []
all_test_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels_tensor = torch.tensor(label_encoder.transform(labels), dtype=torch.long).to(device)
        outputs = model(images)
        preds = torch.max(outputs, 1)[1].cpu().numpy()

        all_test_labels.extend(labels_tensor.cpu().numpy())
        all_test_preds.extend(preds)

test_accuracy = accuracy_score(all_test_labels, all_test_preds)
print(f'Test Accuracy: {test_accuracy:.4f}')
