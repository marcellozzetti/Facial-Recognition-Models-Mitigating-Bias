import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import pandas as pd
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import pre_processing_images

# Definindo o dispositivo (GPU se disponível)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hiperparâmetros
batch_size = 32
learning_rate = 0.001
num_epochs = 25
train_val_split = 0.8  # 80% treino, 20% validação

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
        label = self.labels_df.iloc[idx, 3]  # 'race' is the classe no CSV

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

# Dividindo o dataset em treino e validação
train_size = int(train_val_split * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Criando os DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
dataloaders = {'train': train_loader, 'val': val_loader}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

# Definindo o modelo ResNet50 pré-treinado
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(csv_pd['race'].unique()))  # Número de classes a partir do CSV
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
        cosine = nn.functional.linear(nn.functional.normalize(input), nn.functional.normalize(self.weight))
        theta = torch.acos(cosine.clamp(-1.0, 1.0))
        target_logit = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
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
scaler =  torch.amp.GradScaler(torch.device(device)) if device == torch.device("cuda") else None

print("Scaler: ", scaler)

# Função de treino
def train_model(model, arcface, criterion, optimizer, scheduler, num_epochs=25):
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Modo de treinamento
            else:
                model.eval()   # Modo de validação
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterando sobre os dados
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = label_encoder.transform(labels).to(device)
                #labels = labels.to(device)

                optimizer.zero_grad()

                with autocast(), torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    logits = arcface(outputs, labels)
                    _, preds = torch.max(logits, 1)
                    loss = criterion(logits, labels)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Melhor modelo
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

# Iniciando o treinamento
model_trained = train_model(model, arcface, criterion, optimizer, scheduler, num_epochs=num_epochs)

# Salvando o modelo treinado
torch.save(model_trained.state_dict(), 'fairface_resnet50_arcface.pth')
