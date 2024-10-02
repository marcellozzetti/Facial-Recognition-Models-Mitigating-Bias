import os
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FaceDataset(Dataset):
    def __init__(self, img_paths, labels, img_dir, transform=None, label_encoder=None):
        self.img_paths = img_paths
        self.labels = labels
        self.img_dir = img_dir
        self.transform = transform
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_paths[idx])
        label = self.labels[idx]

        img = cv2.imread(img_name)
        if img is None:
            raise FileNotFoundError(f"Image {img_name} not found")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            #img = Image.fromarray(img)
            img = self.transform(img)
            
        label_index = self.label_encoder.transform([label])[0]

        return img, label_index

def dataset_transformation_train(img):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img)

def dataset_transformation_val(img):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img)
