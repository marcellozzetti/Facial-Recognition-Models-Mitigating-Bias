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
from sklearn.manifold import TSNE
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import torch.utils.checkpoint
from torchvision.models import ResNet50_Weights
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, classification_report, log_loss
from sklearn.preprocessing import LabelEncoder
from face_dataset import FaceDataset, dataset_transformation_train, dataset_transformation_val
from models import LResNet50E_IR, ArcFaceLoss, ArcMarginProduct
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import datetime
from tqdm import tqdm
import pre_processing_images
from gradcam_analysis import generate_grad_cam

# Hyperparameters
BATCH_SIZE = 128
NUM_EPOCHS = 1
TEST_SIZE = 0.1
LEARNING_RATE = 0.01
SCALE = 10
MARGIN = 0.35

experiments = {
    #"CrossEntropyLoss&SGD": {},
    #"ArcFaceLoss&SGD": {},
    #"CrossEntropyLoss&AdamW": {},
    #"ArcFaceLoss&AdamW": {},
    #"CrossEntropyLoss&AdamW&CosineAnnealing": {},
    #"ArcFaceLoss&AdamW&CosineAnnealing": {}, ##FAIL
    #"CrossEntropyLoss&AdamW": {}, #filtered
    #"ArcFaceLoss&AdamW": {}, #filtered
    #"CrossEntropyLoss&AdamW": {}, #dropout .5
    #"ArcFaceLoss&AdamW": {}, #dropout .5
    "CrossEntropyLoss&AdamW": {}, #mais epocas
    #"ArcFaceLoss&AdamW": {}, #mais epocas
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
output_dir = 'output'

print("Step 9 (CNN model): Start")

# Load dataset
csv_pd = pd.read_csv(pre_processing_images.CSV_BALANCED_CONCAT_DATASET_FILE) #CSV_BALANCED_CONCAT_DATASET_FILE || CSV_CONCAT_DATASET_FILTERED_FILE

# Filter class "White" and "Black"
#csv_pd = csv_pd[csv_pd['race'].isin(['White', 'Black'])]

csv_pd = csv_pd.groupby('race').apply(lambda x: x.sample(n=20, random_state=42)).reset_index(drop=True)

label_encoder = LabelEncoder()
label_encoder.fit(csv_pd['race'])
num_classes = len(label_encoder.classes_)

print("Labels: ", label_encoder)

# Create DataSets
X = csv_pd['file']
y = csv_pd['race']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=TEST_SIZE, stratify=y_train, random_state=42)

train_dataset = FaceDataset(X_train.tolist(), y_train.tolist(), pre_processing_images.IMG_PROCESSED_DIR, transform=dataset_transformation_train, label_encoder=label_encoder)
val_dataset = FaceDataset(X_val.tolist(), y_val.tolist(), pre_processing_images.IMG_PROCESSED_DIR, transform=dataset_transformation_val, label_encoder=label_encoder)
test_dataset = FaceDataset(X_test.tolist(), y_test.tolist(), pre_processing_images.IMG_PROCESSED_DIR, transform=dataset_transformation_val, label_encoder=label_encoder)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)

print("Step 9 (CNN model): End")

print("Step 10 (Training execution): Start")

# Training function
def train_model(model, criterion, optimizer, scheduler, scaler, arc_face_margin, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        running_loss = 0.0

        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)
        
            optimizer.zero_grad()
    
            if scaler:
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    if arc_face_margin is not None:
                        outputs = arc_face_margin(outputs, labels)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                if arc_face_margin is not None:
                    outputs = arc_face_margin(outputs, labels)        
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        # Atualizar scheduler no final de cada epoch
        if scheduler:
            scheduler.step()

        overhead = time.time() - start_time
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Overhead: {overhead:.4f}s, Loss: {avg_loss:.4f}')
        print(f"Learning rate: {scheduler.get_last_lr() if scheduler else LEARNING_RATE}")

        # Validação
        model.eval()
        all_labels, all_preds, all_probs = [], [], []
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                if arc_face_margin is not None:
                    outputs = arc_face_margin(outputs, labels)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                probs = F.softmax(outputs, dim=1).cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds)
                all_probs.extend(probs)

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        logloss = log_loss(all_labels, np.array(all_probs))

        # Salvando métricas
        train_losses.append(avg_loss)
        accuracies.append(accuracy)
        precisions.append(precision)
        log_losses.append(logloss)
        
        print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {accuracy:.4f}, "
              f"Precision: {precision:.4f}, Log Loss: {logloss:.4f}")

    return model

# Evaluate function
def evaluate_model(model, test_loader, criterion, arc_face_margin, label_encoder):
        model.eval()
        all_labels = []
        all_preds = []
        all_probs = []
        incorrect_indices = []
        epoch_loss = 0.0
        seen_classes = set()  # To track which classes we've added incorrect indices for
    
        with torch.no_grad():
            for idx, (images, labels) in enumerate(tqdm(test_loader)):
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(images)
                
                if arc_face_margin is not None:
                    outputs = arc_face_margin(outputs, labels)
                    
                loss = criterion(outputs, labels)
                epoch_loss += loss.item()
            
                # Get probabilities
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                
                # Get predicted class (argmax)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds)
                all_probs.extend(probs)

                # Check incorrect predictions
                incorrect_preds = np.where(preds != labels.cpu().numpy())[0]
                
                # For each incorrect prediction, only add the first occurrence of the class
                for i in incorrect_preds:
                    pred_class = preds[i]
                    # Check if we already added an incorrect index for this class
                    if pred_class not in seen_classes:
                        incorrect_indices.append(idx * BATCH_SIZE + i)
                        seen_classes.add(pred_class)
    
        # Generate images to Grad-CAM
        generate_grad_cam(model, images, labels, incorrect_indices, label_encoder)
            
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
    
def evaluate_model_with_tsne(model, test_loader, criterion, arc_face_margin, label_encoder):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    incorrect_indices = []
    epoch_loss = 0.0
    seen_classes = set()  # To track which classes we've added incorrect indices for
    
    embeddings = []  # Lista para armazenar as embeddings
    labels_list = []  # Lista para armazenar as labels

    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            if arc_face_margin is not None:
                outputs = arc_face_margin(outputs, labels)
                
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
        
            # Get probabilities
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            
            # Get predicted class (argmax)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
        
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

            # Check incorrect predictions
            incorrect_preds = np.where(preds != labels.cpu().numpy())[0]
            
            # For each incorrect prediction, only add the first occurrence of the class
            for i in incorrect_preds:
                pred_class = preds[i]
                if pred_class not in seen_classes:
                    incorrect_indices.append(idx * len(images) + i)  # Usar o comprimento das imagens para o índice
                    seen_classes.add(pred_class)
            
            # Armazenar embeddings e labels para a visualização t-SNE
            embeddings.append(outputs.cpu().numpy())  # Guardar as saídas do modelo
            labels_list.append(labels.cpu().numpy())  # Guardar as labels

    # Concatenar as embeddings e labels
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot the t-SNE results
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        indices = labels == label
        plt.scatter(
            embeddings_2d[indices, 0], 
            embeddings_2d[indices, 1], 
            label=label_encoder.inverse_transform([label])[0], 
            alpha=0.7
        )

    plt.title("t-SNE Visualization of Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the t-SNE plot
    save_path = f'{output_dir}/tsne_visualization_{timestamp}.png'
    plt.savefig(save_path)
    print(f"t-SNE visualization saved to {save_path}")
    plt.show()

    # Generate images to Grad-CAM
    generate_grad_cam(model, images, labels, incorrect_indices, label_encoder)

    # Calculando métricas
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Garantir que all_probs seja um numpy array e calcular o log loss
    all_probs = np.array(all_probs)
    logloss = log_loss(all_labels, all_probs)

    print(f'Test Accuracy: {accuracy:.4f}, Test Precision: {precision:.4f}, Test Log Loss: {logloss:.4f}')
    
    # Plot confusion matrix
    confusion_mtx = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx, annot=True, fmt="d", cmap="Blues")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{output_dir}/confusion_mtx_{timestamp}.png')
    plt.show()
    
    # Gerar relatório de classificação
    report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_)
    print("\nClassification Report:\n", report)
    
    # Salvar o relatório de classificação em um arquivo
    report_filename = f'{output_dir}/classification_report_{timestamp}.txt'
    with open(report_filename, 'w') as f:
        f.write(report)
    
    print(f"\nClassification report saved to {report_filename}")
    
# Loop for each experiemnt
for exp in experiments.keys():

    print(f'Step 10 (Training execution): Start - {exp}')

    # Initializing metrics lists
    train_losses, val_losses, accuracies, precisions, log_losses = [], [], [], [], []
    arc_margin = None

    # Initialize model, criterion, and optimizer
    scaler =  torch.amp.GradScaler(torch.device(device)) if device == torch.device("cuda") else None
    
    if "CrossEntropyLoss" in exp:
        model = LResNet50E_IR(num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
    
    else:
        model = LResNet50E_IR(512).to(device)
        arc_margin = ArcMarginProduct(512, num_classes, SCALE, MARGIN).to(device)
        criterion = ArcFaceLoss(margin=MARGIN, scale=SCALE).to(device)

    model = nn.DataParallel(model)

    #optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0005)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, epochs=NUM_EPOCHS, steps_per_epoch=len(train_loader))
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=2)
    
    model = train_model(model, criterion, optimizer, scheduler, scaler, arc_margin, NUM_EPOCHS)
    
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

    evaluate_model(model, test_loader, criterion, arc_margin, label_encoder)
    #evaluate_model_with_tsne(model, test_loader, criterion, arc_margin, label_encoder)

    print(f'Step 13 (Testing): End - {exp}')
