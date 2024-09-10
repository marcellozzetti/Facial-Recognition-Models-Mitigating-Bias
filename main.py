#1. Realizar a instalação de bibliotecas adicionais e import no Python

print("Step 1 (Imports): Starting")

import pandas as pd
import seaborn as sns
import numpy as npy

import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objs as go

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler

from torch.amp import GradScaler, autocast

from sklearn.metrics import accuracy_score, precision_score, log_loss
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle
from scipy.spatial import distance

from mtcnn.mtcnn import MTCNN

from PIL import Image

import zipfile
import requests
import time
import math
import io
import os
import ssl
import cv2
import gc

print("Step 1 (Imports): End")

#2. Definir variaveis globais
print("Step 2 (Global Variables): Start")

# To download external packages
ssl._create_default_https_context = ssl._create_unverified_context

# Main paths
base_dir = '/home/ubuntu/projects'

img_base_dir = base_dir + '/fairface/dataset/'
img_processed_dir = base_dir + '/fairface/dataset/output/processed_images/'
csv_concat_dataset_file = base_dir + '/fairface/dataset/fairface_concat_dataset.csv'
csv_balanced_concat_dataset_file = base_dir + '/fairface/dataset/fairface_balanced_concat_dataset.csv'
model_fairface_file = base_dir + '/fairface/dataset/output/fairface_model.pth'

# From GitHUB
csv_train_pd = pd.read_csv('https://raw.githubusercontent.com/marcellozzetti/Facial-Recognition-Models-Mitigating-Bias/main/dataset/fairface_label_train.csv')
csv_val_pd = pd.read_csv('https://raw.githubusercontent.com/marcellozzetti/Facial-Recognition-Models-Mitigating-Bias/main/dataset/fairface_label_val.csv')
csv_train_lab_pd = pd.read_csv('https://raw.githubusercontent.com/marcellozzetti/Facial-Recognition-Models-Mitigating-Bias/main/dataset/fairface_label_train-lab.csv')
zip_dataset_file = 'https://dataset-fairface.s3.us-east-1.amazonaws.com/dataset/fairface-img-margin125-trainval.zip?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEOT%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIDFgdw8tHZTua3qFGVv8JTTfKYalK0bZnI%2BsvTnb6yJFAiAK7ZHVBULKsASufMewFxcUhuZomMuNAIuzcGUQ1csdUCrtAgj8%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAAaDDIwNTkzMDYyMDExMyIMLLt%2BKxD3qeC1ojimKsECjCttAYHj%2B%2FvuAwYMF%2BpsviCREr%2FLkwOtHce79dsOwSvmfAQCTexen45eh4R%2BJzk0%2FJtifcCOjf4rst%2Bf8WXRrMeyx8H%2FP4eSQVZJHs0sL0Cx1jvecrLEkTuE5bUZiEeje68qKG8DFAFm2G3vocVBL%2BZ0eaBqFAZZLMVG8qGkLsuleXd0wFnMpnJg8Ij8BdvwYd68lX3uYQ3e2jQzIFNuM63ije6MksZ9jHX%2BFdywtLECkSyKKniZiuSIRJ8b8ZXxK6P6X04kpqMLfDlmcW%2BHIhoato8CxuqGeJ2VaNMyGAc7rx99psBHneLtzQJh8%2FcZlnbZIGekpPm9WN5H6ucNmc8rEaGBYf9udWDsAB2TE7XkFA16x1I4GuFVW9pNErtRW%2F4S5BB8VHiO3uGsEehTvbAWJI7pVRjV0l%2B4gE04vfAAMOib57YGOrQCA8UKN6fxDia6J3h%2FySRSHNn6yHrATB2OtAMFllRwTRAdXylnVHZ4MOPwXwp4v3wCDKFylsuObIvXj8Nl2TERAFOVpeMEEmA3r%2B3ZRaJ1x%2BqMicolV5myy6uewVVeyicpLkLWSZrSYqt4QtniRVx%2F9eujH8I3JoD7hV1m%2FiGvJY6nr6hMnueVZY5A1xsHSP7XlXfpAYoGFgi0H26WWAYGKsnA%2Bq5MEudIYEKUHLUafPO3mZWd55ACtJVIzLHseVX9jq6jKLxL%2Bcp2icLdZNzAztoZCjNRsP8iOJ4ss2R1VLPhKxuwiSelZolAGkoJ2Yino%2Fasg8cVtT0Dy%2BmuFfzBbWfhvqc0o%2FYIB9nehAr1oo66I%2BfZwi0ZV65ipccsqwhetpQwleLF2QXPyW7SL0YBWMUQsNs%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240906T031808Z&X-Amz-SignedHeaders=host&X-Amz-Expires=43200&X-Amz-Credential=ASIAS74TL4TIQTEXWCAZ%2F20240906%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=b3f58c463c45dad9758ccb84538088443431917890f0e795c8afabe797204270'

# Initialize the MTCNN detector
detector = MTCNN()

# Check if Cuda is available
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
print("device: ", device)

print("Step 2 (Global Variables): End")

#3. Juntar os datasets (treino e validação)
print("Step 3 (Join Dataset): Start")

# Concat two dataframes
csv_concatenated = pd.concat([csv_train_pd, csv_val_pd])

# Checking if exists
os.makedirs(img_base_dir, exist_ok=True)

csv_concatenated.to_csv(csv_concat_dataset_file, index=False)

print("Step 3 (Join Dataset): End")

#4. Importar o dataset de imagens com a diversidade de classes generio, raca, idade
print("Step 4 (Import dataSet): Start")

# Load dataset
df = pd.read_csv(csv_concat_dataset_file)

# Dataset size
print("Dataset size:")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# Statistical overview
print("\nStatistical Overview:")
print(df.describe())

# Checking null values
print("\nChecking null values:")
print(df.isnull().sum())

# Using plotly to create interactive dashboards
class_columns = ['age', 'gender', 'race']

for column in class_columns:
    print(f"\nColumn Class Distribuition '{column}':")
    class_distribution = df[column].value_counts()
    print(class_distribution)

    # Ploting class distribution using seaborn
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=column)
    plt.title(f'Column Class Distribuition: {column}')
    plt.show()

fig = go.Figure()

# Graphs for each different class
for column in class_columns:
    fig.add_trace(go.Histogram(x=df[column], name=f'Histogram {column}'))

# Dashboard layout
fig.update_layout(title='Analytical Class Dashboard',
                  xaxis_title='Class',
                  yaxis_title='Count',
                  barmode='overlay')

fig.show()

print("Step 4 (Import dataSet): End")

#5. Rebalancear o dataset
print("Step 5 (Rebalance Dataset): Start")

# Main class to be balanced
class_column = 'race'

# Count samples per class
class_counts = df[class_column].value_counts()
print("Sample count per class before balancing:")
print(class_counts)

# Identify the minority class
minority_class = class_counts.idxmin()
min_count = class_counts.min()

# Create a new empty DataFrame to store balanced samples
balanced_df = pd.DataFrame()

# Perform undersampling of majority classes
for class_ in class_counts.index:
    class_df = df[df[class_column] == class_]
    if len(class_df) > min_count:
        # Perform undersampling of the class
        class_df = resample(class_df, replace=False, n_samples=min_count, random_state=42)
    balanced_df = pd.concat([balanced_df, class_df])

# Check the new counts
balanced_class_counts = balanced_df[class_column].value_counts()
print("\nSample count per class after balancing:")
print(balanced_class_counts)

# Save the new dataset
balanced_df.to_csv(csv_balanced_concat_dataset_file, index=False)

# Ploting
fig = go.Figure()
for column in class_columns:
    fig.add_trace(go.Histogram(x=balanced_df[column], name=f'Histogram {column}'))

# New dashboard layout
fig.update_layout(title='Analytical Class Dashboard',
                  xaxis_title='Classes',
                  yaxis_title='Contagem',
                  barmode='overlay')

fig.show()

print("Step 5 (Rebalance Dataset): End")

#6. Definição das funções de tratamento de imagens e faces
print("Step 6 (Imagens Functions): Start")

# This function crop the given face in img based on given coordinates
def cropping_procedure(img, x, y, width, height):
    border = 60
    return img[y-border:y+height+border, x-border:x+width+border]

# This function generate a new rotated image
def rotate_procedure(img, angle):
    center = (img.shape[1] // 2, img.shape[0] // 2)
    matriz_rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, matriz_rotation, (img.shape[1], img.shape[0]))

# This function aligns the given face in img based on left and right eye coordinates
def alignment_procedure(img, left_eye, right_eye):
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    delta_x = right_eye_x - left_eye_x
    delta_y = right_eye_y - left_eye_y
    angle = npy.degrees(npy.arctan2(delta_y, delta_x))

    eyes_center = ((left_eye_x + right_eye_x) // 2, (left_eye_y + right_eye_y) // 2)
    matriz_rotation = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
    rotated_img = cv2.warpAffine(img, matriz_rotation, (img.shape[1], img.shape[0]))

    return rotated_img

# Function to detect faces and draw bounding boxes
def draw_bounding_procedure(img):

    #Ploting
    plt.imshow(img)
    ax = plt.gca()

    detect_faces = detector.detect_faces(img)

    # If no face is detected, return
    if len(detect_faces) == 0:
        print("No face detected.")
        return

    #Considered the first face detected
    keypoints = detect_faces[0]['keypoints']
    x, y, width, height = detect_faces[0]['box']

    rect = Rectangle((x, y), width, height, fill=False, color='red')
    ax.add_patch(rect)

    # Keypoints (eyes, nose, mouth)
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']
    nose = keypoints['nose']
    mouth_left = keypoints['mouth_left']
    mouth_right = keypoints['mouth_right']

    # Draw circles on the keypoints
    ax.add_patch(Circle(left_eye, radius=2, color='blue'))
    ax.add_patch(Circle(right_eye, radius=2, color='blue'))
    ax.add_patch(Circle(nose, radius=2, color='green'))
    ax.add_patch(Circle(mouth_left, radius=2, color='orange'))
    ax.add_patch(Circle(mouth_right, radius=2, color='orange'))

    # Add labels to the keypoints
    ax.text(left_eye[0], left_eye[1], 'Left Eye', fontsize=8, color='blue', verticalalignment='bottom')
    ax.text(right_eye[0], right_eye[1], 'Right Eye', fontsize=8, color='blue', verticalalignment='bottom')
    ax.text(nose[0], nose[1], 'Nose', fontsize=8, color='green', verticalalignment='bottom')
    ax.text(mouth_left[0], mouth_left[1], 'Mouth Left', fontsize=8, color='orange', verticalalignment='bottom')
    ax.text(mouth_right[0], mouth_right[1], 'Mouth Right', fontsize=8, color='orange', verticalalignment='bottom')

    plt.axis('off')
    plt.show()

# Function to detect and perform face adjustments
def detect_and_adjust_faces(img, img_name, save_dir=None, draw_bounding=False):

    # Transform to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    detect_faces = detector.detect_faces(img_rgb)

    # If no face is detected, return
    if len(detect_faces) == 0:
        print("No face detected: {}".format(img_name))
        return False

    #Considered the first face detected
    keypoints = detect_faces[0]['keypoints']
    x, y, width, height = detect_faces[0]['box']
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']

    aligned_face = alignment_procedure(img_rgb, left_eye, right_eye)
    threated_face = cropping_procedure(aligned_face, x, y, width, height)

    if draw_bounding:
      draw_bounding_procedure(threated_face)

    # Redimensionar a face e verificar se a imagem não está vazia
    try:
        threated_face = cv2.resize(threated_face, (224, 224), interpolation=cv2.INTER_AREA)
    except cv2.error as e:
        print(f"Failed to resize image {img_name}. Skipping.")
        return False

    threated_face = cv2.cvtColor(threated_face, cv2.COLOR_BGR2RGB)

    if save_dir:
      os.makedirs(save_dir, exist_ok=True)
      save_path = os.path.join(save_dir, img_name)
      print("Save threated image:", save_path)
      cv2.imwrite(save_path, threated_face)

    return True

print("Step 6 (Imagens Functions): End")

#9. Executar os ajustes e das imagens do dataset
print("Step 9 (Images Adjustments): Start")

# Getting the file name
img_names = csv_train_lab_pd['file']

# Call the function to perform the adjustments
if False:
    for img_name in img_names:
        img_path = os.path.join(img_base_dir, img_name)
        img = cv2.imread(img_path)
        detect_and_adjust_faces(img, img_name, img_processed_dir, False)

print("Step 9 (Images Adjustments): End")

#10. Definir o modelo de treinamento

print("Step 10 (CNN model): Start")

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

            #results = detector.detect_faces(img)

            #if len(results) == 0:
            #    raise ValueError("No face detected")

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

# Load the dataset
dataset = FaceDataset(csv_train_lab_pd, img_processed_dir, transform=transform)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

def collate_fn(batch):
    batch = [item for item in batch if item[0] is not None and item[1] is not None]
    if len(batch) == 0:
        return torch.empty(0), torch.empty(0)
    return torch.utils.data.dataloader.default_collate(batch)

label_encoder = LabelEncoder()
label_encoder.fit(csv_train_lab_pd['race'])

#class_counts = len(label_encoder.classes_)
#class_weights = 1.0 / class_counts.float()
#sample_weights = class_weights[torch.tensor(csv_train_lab_pd['race'].values)]
#sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
#train_loader = DataLoader(train_dataset, batch_size=4, sampler=sampler, collate_fn=collate_fn)

# Create DataLoaders using a filter function: collate_fn
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2, collate_fn=collate_fn)

val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2, collate_fn=collate_fn)

# Define the model (LResNet50E-IR, a modified ResNet50 for ArcFace)
class LResNet50E_IR(nn.Module):
    def __init__(self, num_classes=len(label_encoder.classes_)):
        super(LResNet50E_IR, self).__init__()
        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.dropout = nn.Dropout(p=0.5)  # Adding dropout to reduce overfitting
        self.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.backbone.fc = self.fc

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        return x
    
# Adjustments learning Rate
def adjust_learning_rate(optimizer, epoch):
    if epoch < 10:
        lr = 0.01
    elif epoch < 15:
        lr = 0.001
    else:
        lr = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Initialize model, criterion, and optimizer
model = LResNet50E_IR().to(device)
model = nn.DataParallel(model)  # Paraleliza o modelo entre várias GPUs

#criterion = ArcFaceLoss()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

#optimizer = optim.Adam(model.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

def softmax(x):
    exp_x = npy.exp(x - npy.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

print("Step 10 (CNN model): End")

#11. Treinar e extrair as métricas

print("Step 11 (Training execution): Start")

num_epochs = 50

# General Metrics
train_losses = []
accuracies = []
precisions = []
log_losses = []

scaler = GradScaler()
accumulation_steps = 4

for epoch in range(num_epochs):
    adjust_learning_rate(optimizer, epoch)

    model.train()
    start_time = time.time()
    
    for images, labels in train_loader:
        images = images.to(device)

        labels_tensor = torch.tensor(label_encoder.transform(labels)).to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels_tensor)
    
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #scheduler.step()

    overhead = time.time() - start_time

    # Metrics for training step
    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'Overhead: {overhead:.4f}s')

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

            #print("Outputs val:", outputs)
            #print("Labels val:", labels_tensor)
            #print("probs val:", probs)
            #print("preds val:", preds)

            all_labels.extend(labels_tensor)
            all_preds.extend(preds)
            all_probs.extend(probs)

    # Numpy conversion 
    #all_labels = npy.array(all_labels)
    #all_preds = npy.array(all_preds)
    #all_probs = npy.array(all_probs)
    all_labels = [label.item() for label in all_labels]

    #print("all_labels", all_labels)
    #print("all_preds", all_preds)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)

    all_probs = softmax(all_probs)
    logloss = log_loss(all_labels, all_probs)

    scheduler.step(epoch_loss)

    # Armazenando as métricas
    train_losses.append(epoch_loss / len(val_loader))
    accuracies.append(accuracy)
    precisions.append(precision)
    log_losses.append(logloss)

    # Metrics for validation step
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}, '
          f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Log Loss: {logloss:.4f}')

    # Resources optimization
    del images, labels, outputs, loss
    gc.collect()
    torch.cuda.empty_cache()

torch.save(model.state_dict(), model_fairface_file)
print('Finished Training and Model Saved')

print("train_losses", train_losses)
print("accuracies", accuracies)
print("precisions", precisions)
print("log_losses", log_losses)

# Ploting general metrics
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
plt.show()

print("Step 11 (Training execution): End")