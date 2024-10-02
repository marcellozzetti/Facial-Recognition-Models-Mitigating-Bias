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
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, classification_report, log_loss
from sklearn.preprocessing import LabelEncoder
from face_dataset import FaceDataset, dataset_transformation
from models import LResNet50E_IR, ArcFaceLoss
from sklearn.utils.class_weight import compute_class_weight
import pre_processing_images
import datetime
from tqdm import tqdm

# Hyperparameters
BATCH_SIZE = 128
NUM_EPOCHS = 8
TRAIN_VAL_SPLIT = 0.8
VAL_VAL_SPLIT = 0.1
LEARNING_RATE = 0.001

experiments = {
    "CrossEntropyLoss&SGD&.5DROPOUT": {},
#    "ArcFaceLoss&SGD.5DROPOUT": {},
}

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clean memory
def clean_memory():
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()

clean_memory()

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')

print("Step 9 (CNN model): Start")

# Load dataset
csv_pd = pd.read_csv(pre_processing_images.CSV_BALANCED_CONCAT_DATASET_FILE) #CSV_BALANCED_CONCAT_DATASET_FILE || CSV_CONCAT_DATASET_FILTERED_FILE
dataset = FaceDataset(csv_pd, pre_processing_images.IMG_PROCESSED_DIR, transform=dataset_transformation, label_encoder=label_encoder)

label_encoder = LabelEncoder()
label_encoder.fit(csv_pd['race'])
num_classes = len(label_encoder.classes_)

# Split dataset
train_size = int(TRAIN_VAL_SPLIT * len(dataset))
val_size = int(VAL_VAL_SPLIT * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)

# Initialize model, criterion, and optimizer
model = LResNet50E_IR(num_classes).to(device)
model = nn.DataParallel(model)
scaler =  torch.amp.GradScaler(torch.device(device)) if device == torch.device("cuda") else None

print("Step 9 (CNN model): End")

print("Step 10 (Training execution): Start")

# Training function
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        
        for images, labels in tqdm(train_loader):    
            images = images.to(device)
            labels_tensor = torch.tensor(label_encoder.transform(labels)).to(device)
            optimizer.zero_grad()
    
            if scaler:
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels_tensor)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels_tensor)
                loss.backward()
                optimizer.step()
    
            scheduler.step()
    
        overhead = time.time() - start_time
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Overhead: {overhead:.4f}s')
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Learning rate: {scheduler.get_last_lr()}")

        # Validation
        model.eval()
        all_labels = []
        all_preds = []
        all_probs = []
        epoch_loss = 0.0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader):
                images = images.to(device)
                print("labels train", labels)
                labels_tensor = torch.tensor(label_encoder.transform(labels)).to(device)
                print("labels_tensor train", labels_tensor)
                
                # Forward pass
                outputs = model(images)

                loss = criterion(outputs, labels_tensor)
                epoch_loss += loss.item()
        
                # Get probabilities
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                
                # Get predicted class (argmax)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
        
                all_labels.extend(labels_tensor.cpu().numpy())
                all_preds.extend(preds)
                all_probs.extend(probs)
        
        # Calculating metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # Ensure all_probs is an array before calculating log_loss
        all_probs = np.array(all_probs)
        logloss = log_loss(all_labels, all_probs)
        
        # Append metrics for tracking
        train_losses.append(epoch_loss / len(val_loader))
        accuracies.append(accuracy)
        precisions.append(precision)
        log_losses.append(logloss)
        
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}, '
              f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Log Loss: {logloss:.4f}')

    return model

# Loop for each experiemnt
for exp in experiments.keys():

    print(f'Step 10 (Training execution): Start - {exp}')

    # Initializing metrics lists
    train_losses, val_losses, accuracies, precisions, log_losses = [], [], [], [], []
    
    if "CrossEntropyLoss" in exp:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = ArcFaceLoss().to(device)

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, epochs=NUM_EPOCHS, steps_per_epoch=len(train_loader))
    
    model = train_model(model, criterion, optimizer, scheduler, NUM_EPOCHS)
    
    torch.save(model.state_dict(), os.path.join(pre_processing_images.BASE_DIR, f'fairface/dataset/output/fairface_model_{exp}_{timestamp}.pth'))

    print(f'Finished Training and Model Saved - {exp}')

    print(f'Step 11 (Training execution): End - {exp}')
    
    
    print(f'Step 12 (Plotting execution): Start - {exp}')

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
    plt.savefig(f'output/training_metrics_{exp}_{timestamp}.png')
    plt.show()
    plt.close()

    print(f'Step 12 (Plotting execution): End - {exp}')


    print(f'Step 13 (Testing): Start - {exp}')
    
    def evaluate_model(model, test_loader, criterion, label_encoder):
        # Ensure the model is in evaluation mode
        model.eval()
        all_labels = []
        all_preds = []
        all_probs = []
        epoch_loss = 0.0
    
        with torch.no_grad():
            for images, labels in tqdm(test_loader):
                images = images.to(device)
                
                # Apply label encoding and convert to tensor
                labels_tensor = torch.tensor(label_encoder.transform(labels)).to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels_tensor)
                epoch_loss += loss.item()
            
                # Get probabilities
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                
                # Get predicted class (argmax)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
                all_labels.extend(labels_tensor.cpu().numpy())
                all_preds.extend(preds)
                all_probs.extend(probs)
            
        # Calculating metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # Ensure all_probs is a numpy array and calculate log loss
        all_probs = np.array(all_probs)
        logloss = log_loss(all_labels, all_probs)
    
        print(f'Test Accuracy: {accuracy:.4f}, Test Precision: {precision:.4f}, Test Log Loss: {logloss:.4f}')
    
        # Plot confusion matrix
        confusion_mtx = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_mtx, annot=True, fmt="d", cmap="Blues")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'output/confusion_tx_{exp}_{timestamp}.png')
        plt.show()
        
        # Generate classification report
        report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_)
        print("\nClassification Report:\n", report)
    
        # Save the classification report to a file
        report_filename = f'output/classification_report_{exp}_{timestamp}.txt'
        with open(report_filename, 'w') as f:
            f.write(report)
        
        print(f"\nClassification report saved to {report_filename}")
    
    evaluate_model(model, test_loader, criterion, label_encoder)

    print(f'Step 13 (Testing): End - {exp}')
