import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, log_loss, accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from torchvision.models import ResNet50_Weights
import pre_processing_images

# Definindo o dispositivo (GPU se disponível)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hiperparâmetros
batch_size = 64
learning_rate = 0.001
num_epochs = 25
train_val_split = 0.7  # 70% treino, 15% validação, 15% teste

# Dataset customizado usando CSV para as labels
class FaceDataset(Dataset):
    def __init__(self, csv_pd, img_dir, transform=None):
        self.labels_df = csv_pd
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.labels_df.iloc[idx, 0])
        label = self.labels_df.iloc[idx, 3]  # Coluna com o rótulo 'race'

        img = cv2.imread(img_name)
        if img is None:
            raise FileNotFoundError(f"Image {img_name} not found")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = Image.fromarray(img)
            img = self.transform(img)
        return img, label

# Transformações e Data Augmentation
transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Carregando o dataset do CSV
csv_pd = pd.read_csv(pre_processing_images.CSV_BALANCED_CONCAT_DATASET_FILE)
dataset = FaceDataset(csv_pd, pre_processing_images.IMG_PROCESSED_DIR, transform=transform)

# Dividindo o dataset em treino, validação e teste
train_size = int(train_val_split * len(dataset))
val_test_split = (len(dataset) - train_size) // 2
val_size = val_test_split
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Criando os DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}

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
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # Normalizando entrada e pesos
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        theta = torch.acos(cosine.clamp(-1.0, 1.0))
        target_logit = torch.cos(theta + self.m)

        # One-hot encoding dos rótulos
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Combina logits com margem e ajusta escala
        output = (one_hot * target_logit) + ((1.0 - one_hot) * cosine)
        output *= self.s

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

# Inicializando listas para armazenar as métricas
train_losses, val_losses, accuracies, precisions, log_losses = [], [], [], [], []

# Função de treino
def train_model(model, arcface, criterion, optimizer, scheduler, num_epochs=25):
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
    
                with torch.amp.autocast("cuda"), torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    logits = arcface(outputs, labels_tensor)
    
                    probs = F.softmax(logits, dim=1)
                    _, preds = torch.max(logits, 1)
                    loss = criterion(logits, labels_tensor)
    
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

    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model

# Treinando o modelo
model = train_model(model, arcface, criterion, optimizer, scheduler, num_epochs=num_epochs)

# Avaliando o modelo no conjunto de teste
def evaluate_model(model, arcface, test_loader, criterion, device, label_encoder):
    model.eval()
    test_loss = 0.0
    running_corrects = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels_tensor = torch.tensor(label_encoder.transform(labels)).to(device)

            outputs = model(inputs)
            logits = arcface(outputs, labels_tensor)

            # Calcular as probabilidades e previsões
            probs = F.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)

            loss = criterion(logits, labels_tensor)

            test_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels_tensor.data)

            # Coletar rótulos verdadeiros e previsões
            all_labels.extend(labels_tensor.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Cálculo das métricas de desempenho
    test_loss /= len(test_loader.dataset)
    test_acc = running_corrects.double() / len(test_loader.dataset)
    
    precision = precision_score(all_labels, all_preds, average='weighted')
    confusion_mtx = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_)

    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Acc: {test_acc:.4f}')
    print(f'Precision: {precision:.4f}')
    
    # Exibindo a matriz de confusão
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()
    
    print("\nClassification Report:\n", report)

# Avaliando o modelo no conjunto de teste
evaluate_model(model, arcface, test_loader, criterion, device, label_encoder)
