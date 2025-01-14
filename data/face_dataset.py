import os
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class FaceDataset(Dataset):
    """
    Custom Dataset for handling image data and labels stored in a directory structure.
    """
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
            img = self.transform(img)
            
        label_index = self.label_encoder.transform([label])[0]

        return img, label_index

def load_data(data_config):
    """
    Loads training, validation, and testing datasets using FaceDataset.
    """
    # Define data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((data_config['input_size'], data_config['input_size'])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((data_config['input_size'], data_config['input_size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((data_config['input_size'], data_config['input_size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Create datasets
    datasets = {
        'train': FaceDataset(data_config['train_dir'], transform=data_transforms['train']),
        'val': FaceDataset(data_config['val_dir'], transform=data_transforms['val']),
        'test': FaceDataset(data_config['test_dir'], transform=data_transforms['test'])
    }

    # Create dataloaders
    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=data_config['batch_size'], shuffle=True, num_workers=data_config['num_workers']),
        'val': DataLoader(datasets['val'], batch_size=data_config['batch_size'], shuffle=False, num_workers=data_config['num_workers']),
        'test': DataLoader(datasets['test'], batch_size=data_config['batch_size'], shuffle=False, num_workers=data_config['num_workers'])
    }

    return dataloaders, datasets['train'].class_to_idx