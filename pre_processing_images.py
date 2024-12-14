print("Step 1 (Imports): Start")

# Imports
import os
import ssl
import gc
import time
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
import plotly.express as px
import plotly.graph_objs as go

print("Step 1 (Imports): End")

print("Step 2 (Global Variables): Start")
# Constants
BASE_DIR = '/home/azureuser/cloudfiles/code/Users/marcello.ozzetti'
IMG_BASE_DIR = os.path.join(BASE_DIR, 'fairface/dataset/')
IMG_PROCESSED_DIR = os.path.join(BASE_DIR, 'fairface/dataset/output/processed_images/')
CSV_CONCAT_DATASET_FILE = os.path.join(BASE_DIR, 'fairface/dataset/fairface_concat_dataset.csv')
CSV_CONCAT_DATASET_FILTERED_FILE = os.path.join(BASE_DIR, 'Facial-Recognition-Models-Mitigating-Bias/dataset/fairface_concat_filtered_dataset.csv')
CSV_BALANCED_CONCAT_DATASET_FILE = os.path.join(BASE_DIR, 'fairface/dataset/fairface_balanced_concat_dataset.csv')

PERFORM_ADJUSTMENTS = False
MAX_SAMPLES = 5000

# To download external packages
ssl._create_default_https_context = ssl._create_unverified_context

# Initialize the MTCNN detector
detector = MTCNN(cuda=False)

# Check if Cuda is available
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
print("device: ", device)

# Load datasets from GitHub
csv_train_pd = pd.read_csv('https://raw.githubusercontent.com/marcellozzetti/Facial-Recognition-Models-Mitigating-Bias/main/dataset/fairface_label_train.csv')
csv_val_pd = pd.read_csv('https://raw.githubusercontent.com/marcellozzetti/Facial-Recognition-Models-Mitigating-Bias/main/dataset/fairface_label_val.csv')
csv_train_lab_pd = pd.read_csv('https://raw.githubusercontent.com/marcellozzetti/Facial-Recognition-Models-Mitigating-Bias/main/dataset/fairface_label_train-lab.csv')
zip_dataset_file = 'https://dataset-fairface.s3.us-east-1.amazonaws.com/dataset/fairface-img-margin125-trainval.zip?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEOT%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIDFgdw8tHZTua3qFGVv8JTTfKYalK0bZnI%2BsvTnb6yJFAiAK7ZHVBULKsASufMewFxcUhuZomMuNAIuzcGUQ1csdUCrtAgj8%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAAaDDIwNTkzMDYyMDExMyIMLLt%2BKxD3qeC1ojimKsECjCttAYHj%2B%2FvuAwYMF%2BpsviCREr%2FLkwOtHce79dsOwSvmfAQCTexen45eh4R%2BJzk0%2FJtifcCOjf4rst%2Bf8WXRrMeyx8H%2FP4eSQVZJHs0sL0Cx1jvecrLEkTuE5bUZiEeje68qKG8DFAFm2G3vocVBL%2BZ0eaBqFAZZLMVG8qGkLsuleXd0wFnMpnJg8Ij8BdvwYd68lX3uYQ3e2jQzIFNuM63ije6MksZ9jHX%2BFdywtLECkSyKKniZiuSIRJ8b8ZXxK6P6X04kpqMLfDlmcW%2BHIhoato8CxuqGeJ2VaNMyGAc7rx99psBHneLtzQJh8%2FcZlnbZIGekpPm9WN5H6ucNmc8rEaGBYf9udWDsAB2TE7XkFA16x1I4GuFVW9pNErtRW%2F4S5BB8VHiO3uGsEehTvbAWJI7pVRjV0l%2B4gE04vfAAMOib57YGOrQCA8UKN6fxDia6J3h%2FySRSHNn6yHrATB2OtAMFllRwTRAdXylnVHZ4MOPwXwp4v3wCDKFylsuObIvXj8Nl2TERAFOVpeMEEmA3r%2B3ZRaJ1x%2BqMicolV5myy6uewVVeyicpLkLWSZrSYqt4QtniRVx%2F9eujH8I3JoD7hV1m%2FiGvJY6nr6hMnueVZY5A1xsHSP7XlXfpAYoGFgi0H26WWAYGKsnA%2Bq5MEudIYEKUHLUafPO3mZWd55ACtJVIzLHseVX9jq6jKLxL%2Bcp2icLdZNzAztoZCjNRsP8iOJ4ss2R1VLPhKxuwiSelZolAGkoJ2Yino%2Fasg8cVtT0Dy%2BmuFfzBbWfhvqc0o%2FYIB9nehAr1oo66I%2BfZwi0ZV65ipccsqwhetpQwleLF2QXPyW7SL0YBWMUQsNs%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240906T031808Z&X-Amz-SignedHeaders=host&X-Amz-Expires=43200&X-Amz-Credential=ASIAS74TL4TIQTEXWCAZ%2F20240906%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=b3f58c463c45dad9758ccb84538088443431917890f0e795c8afabe797204270'

print("Step 2 (Global Variables): End")


# Step 3: Join Datasets
print("Step 3 (Join Dataset): Start")
csv_concatenated_pd = pd.concat([csv_train_pd, csv_val_pd])
os.makedirs(IMG_BASE_DIR, exist_ok=True)
csv_concatenated_pd.to_csv(CSV_CONCAT_DATASET_FILE, index=False)
print("Step 3 (Join Dataset): End")

# Step 4: Import Dataset
print("Step 4 (Import DataSet): Start")
csv_concatenated_pd = pd.read_csv(CSV_CONCAT_DATASET_FILE)
print("Dataset size:")
print(f"Rows: {csv_concatenated_pd.shape[0]}")
print(f"Columns: {csv_concatenated_pd.shape[1]}")
print("\nStatistical Overview:")
print(csv_concatenated_pd.describe())
print("\nChecking null values:")
print(csv_concatenated_pd.isnull().sum())

class_columns = ['age', 'gender', 'race']
for column in class_columns:
    print(f"\nColumn Class Distribution '{column}':")
    class_distribution = csv_concatenated_pd[column].value_counts()
    print(class_distribution)
    plt.figure(figsize=(10, 6))
    sns.countplot(data=csv_concatenated_pd, x=column)
    plt.title(f'Column Class Distribution: {column}')
    plt.show()

fig = go.Figure()
for column in class_columns:
    fig.add_trace(go.Histogram(x=csv_concatenated_pd[column], name=f'Histogram {column}'))
fig.update_layout(title='Analytical Class Dashboard', xaxis_title='Class', yaxis_title='Count', barmode='overlay')
fig.show()
print("Step 4 (Import DataSet): End")


print("Step 5 (Image Functions): Start")
def cropping_procedure(img, x, y, width, height):
    """Crop the given face in img based on given coordinates."""
    border = 60
    return img[y-border:y+height+border, x-border:x+width+border]

def rotate_procedure(img, angle):
    """Generate a new rotated image."""
    center = (img.shape[1] // 2, img.shape[0] // 2)
    matrix_rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, matrix_rotation, (img.shape[1], img.shape[0]))

def alignment_procedure(img, left_eye, right_eye):
    """Align the given face in img based on left and right eye coordinates."""
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye
    delta_x = right_eye_x - left_eye_x
    delta_y = right_eye_y - left_eye_y
    angle = np.degrees(np.arctan2(delta_y, delta_x))
    eyes_center = ((left_eye_x + right_eye_x) // 2, (left_eye_y + right_eye_y) // 2)
    matrix_rotation = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
    return cv2.warpAffine(img, matrix_rotation, (img.shape[1], img.shape[0]))

def draw_bounding_procedure(img):
    """Detect faces and draw bounding boxes."""
    plt.imshow(img)
    ax = plt.gca()
    detect_faces = detector.detect_faces(img)
    if len(detect_faces) == 0:
        print("No face detected.")
        return
    keypoints = detect_faces[0]['keypoints']
    x, y, width, height = detect_faces[0]['box']
    rect = Rectangle((x, y), width, height, fill=False, color='red')
    ax.add_patch(rect)
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']
    nose = keypoints['nose']
    mouth_left = keypoints['mouth_left']
    mouth_right = keypoints['mouth_right']
    ax.add_patch(Circle(left_eye, radius=2, color='blue'))
    ax.add_patch(Circle(right_eye, radius=2, color='blue'))
    ax.add_patch(Circle(nose, radius=2, color='green'))
    ax.add_patch(Circle(mouth_left, radius=2, color='orange'))
    ax.add_patch(Circle(mouth_right, radius=2, color='orange'))
    ax.text(left_eye[0], left_eye[1], 'Left Eye', fontsize=8, color='blue', verticalalignment='bottom')
    ax.text(right_eye[0], right_eye[1], 'Right Eye', fontsize=8, color='blue', verticalalignment='bottom')
    ax.text(nose[0], nose[1], 'Nose', fontsize=8, color='green', verticalalignment='bottom')
    ax.text(mouth_left[0], mouth_left[1], 'Mouth Left', fontsize=8, color='orange', verticalalignment='bottom')
    ax.text(mouth_right[0], mouth_right[1], 'Mouth Right', fontsize=8, color='orange', verticalalignment='bottom')
    plt.axis('off')
    plt.show()

def detect_and_adjust_faces(img, img_name, save_dir=None, draw_bounding=False):
    """Detect and perform face adjustments."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detect_faces = detector.detect_faces(img_rgb)
    if len(detect_faces) == 0:
        print("No face detected: {}".format(img_name))
        return False
    keypoints = detect_faces[0]['keypoints']
    x, y, width, height = detect_faces[0]['box']
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']
    aligned_face = alignment_procedure(img_rgb, left_eye, right_eye)
    treated_face = cropping_procedure(aligned_face, x, y, width, height)
    if draw_bounding:
        draw_bounding_procedure(treated_face)
    try:
        treated_face = cv2.resize(treated_face, (224, 224), interpolation=cv2.INTER_AREA)
    except cv2.error as e:
        print(f"Failed to resize image {img_name}. Skipping.")
        return False
    treated_face = cv2.cvtColor(treated_face, cv2.COLOR_BGR2RGB)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, img_name)
        print("Save treated image:", save_path)
        cv2.imwrite(save_path, treated_face)
    return True
print("Step 5 (Image Functions): End")


print("Step 7 (Images Adjustments): Start")
csv_concatenated_pd = pd.read_csv(CSV_CONCAT_DATASET_FILE)
img_names = csv_concatenated_pd['file']
failed_images = []
if PERFORM_ADJUSTMENTS:
    for index, img_name in enumerate(img_names):
        if index >= MAX_SAMPLES:
            break
        img_path = os.path.join(IMG_BASE_DIR, img_name)
        img = cv2.imread(img_path)
        if not detect_and_adjust_faces(img, img_name, IMG_PROCESSED_DIR, False):
            failed_images.append(img_name)
    df_filtered = csv_concatenated_pd[~csv_concatenated_pd['file'].isin(failed_images)]
    df_filtered.to_csv(CSV_CONCAT_DATASET_FILTERED_FILE, index=False)
    print(f"Removed {len(failed_images)} images that failed processing.")
print("Step 7 (Images Adjustments): End")


print("Step 8 (Rebalance Dataset): Start")
csv_concatenated_pd = pd.read_csv(CSV_CONCAT_DATASET_FILTERED_FILE)
class_column = 'race'
class_counts = csv_concatenated_pd[class_column].value_counts()
print("Sample count per class before balancing:")
print(class_counts)
minority_class = class_counts.idxmin()
min_count = class_counts.min()
balanced_df = pd.DataFrame()
for class_ in class_counts.index:
    class_df = csv_concatenated_pd[csv_concatenated_pd[class_column] == class_]
    if len(class_df) > min_count:
        class_df = resample(class_df, replace=False, n_samples=min_count, random_state=42)
    balanced_df = pd.concat([balanced_df, class_df])
balanced_class_counts = balanced_df[class_column].value_counts()
print("\nSample count per class after balancing:")
print(balanced_class_counts)
balanced_df.to_csv(CSV_BALANCED_CONCAT_DATASET_FILE, index=False)
fig = go.Figure()
for column in class_columns:
    fig.add_trace(go.Histogram(x=balanced_df[column], name=f'Histogram {column}'))
fig.update_layout(title='Analytical Class Dashboard', xaxis_title='Classes', yaxis_title='Count', barmode='overlay')
fig.show()
print("Step 8 (Rebalance Dataset): End")
