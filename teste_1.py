import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torch.optim import lr_scheduler
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
from face_dataset import FaceDataset 
import pre_processing_images

# Definindo o dispositivo (GPU se disponível)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hiperparâmetros
batch_size = 64
learning_rate = 0.001
num_epochs = 25
train_val_split = 0.7  # 70% treino, 15% validação, 15% teste

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
val_size = (len(dataset) - train_size) // 2
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Criando os DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}

# Definindo a arquitetura LResNet100E-IR (ResNet100 aprimorada)
class LResNet100E_IR(nn.Module):
    def __init__(self, num_classes):
        super(LResNet100E_IR, self).__init__()
        self.resnet = torch.hub.load('zhanghang1989/ResNeSt', 'resnest100', pretrained=True)  # Exemplo para carregar a ResNet100
        self.resnet.fc = nn.Identity()  # Mantém a saída de 2048
        
        # Definindo a camada ArcFace
        self.arcface = ArcMarginProduct(in_features=2048, out_features=num_classes)

    def forward(self, x, labels):
        features = self.resnet(x)  # Extrair características
        logits = self.arcface(features, labels)  # Calcular os logits com ArcFace
        return logits

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

# Instanciando o modelo
num_classes = len(dataset.classes)
model = LResNet100E_IR(num_classes).to(device)

# Definindo o otimizador e a função de perda
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Definindo o scheduler para ajustar a taxa de aprendizado
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# GradScaler para Mixed Precision
scaler = GradScaler() if device == torch.device("cuda") else None

# Função de treino
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with autocast(device == 'cuda'), torch.set_grad_enabled(phase == 'train'):
                    logits = model(inputs, labels)
                    _, preds = torch.max(logits, 1)

                    loss = criterion(logits, labels)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        scheduler.step()

    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model

# Treinando o modelo
model = train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs)

# Avaliando o modelo no conjunto de teste
def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    running_corrects = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs, labels)
            _, preds = torch.max(logits, 1)

            loss = criterion(logits, labels)

            test_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    test_acc = running_corrects.double() / len(test_loader.dataset)

    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Acc: {test_acc:.4f}')

    precision = precision_score(all_labels, all_preds, average='weighted')
    print(f'Precision: {precision:.4f}')

    confusion_mtx = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx, annot=True, fmt="d", cmap="Blues")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    report = classification_report(all_labels, all_preds, target_names=dataset.classes)
    print("\nClassification Report:\n", report)

# Avaliando o modelo no conjunto de teste
evaluate_model(model, test_loader, criterion)
