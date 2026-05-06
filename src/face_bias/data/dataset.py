"""PyTorch Dataset and DataLoader builders for FairFace.

NOTE: bugs B1-B3 from REVIEW_AND_PLAN.md are preserved here intentionally
during Sprint A (mechanical migration only). They will be fixed in Sprint B.
"""

import logging
import os
from typing import Any

import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class FaceDataset(Dataset):
    """Custom Dataset for handling image data and labels stored in a directory structure."""

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
            logging.error(f"Image {img_name} not found")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)

        label_index = self.label_encoder.transform([label])[0]

        return img, label_index


def setup_dataset(config: dict[str, Any]):
    """Load training, validation, and test datasets using FaceDataset."""
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize((config["image"]["input_size"], config["image"]["input_size"])),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(config["image"]["image_mean"], config["image"]["image_mean"]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((config["image"]["input_size"], config["image"]["input_size"])),
                transforms.ToTensor(),
                transforms.Normalize(config["image"]["image_mean"], config["image"]["image_mean"]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize((config["image"]["input_size"], config["image"]["input_size"])),
                transforms.ToTensor(),
                transforms.Normalize(config["image"]["image_mean"], config["image"]["image_mean"]),
            ]
        ),
    }

    csv_pd = pd.read_csv(config["data"]["dataset_file"])

    label_encoder = LabelEncoder()
    label_encoder.fit(csv_pd["race"])
    num_classes = len(label_encoder.classes_)
    logging.info(f"Labels: {label_encoder} ({num_classes} classes)")

    X = csv_pd["file"]
    y = csv_pd["race"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["training"]["test_size"],
        stratify=y,
        random_state=config["training"]["random_state"],
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=config["training"]["test_size"],
        stratify=y_train,
        random_state=config["training"]["random_state"],
    )

    datasets = {
        "train": FaceDataset(
            X_train.tolist(),
            y_train.tolist(),
            config["data"]["dataset_image_path"],
            transform=data_transforms["train"],
            label_encoder=label_encoder,
        ),
        "val": FaceDataset(
            X_val.tolist(),
            y_val.tolist(),
            config["data"]["dataset_image_path"],
            transform=data_transforms["val"],
            label_encoder=label_encoder,
        ),
        "test": FaceDataset(
            X_test.tolist(),
            y_test.tolist(),
            config["data"]["dataset_image_path"],
            transform=data_transforms["test"],
            label_encoder=label_encoder,
        ),
    }

    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=config["training"]["num_workers"],
            pin_memory=True,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["training"]["num_workers"],
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["training"]["num_workers"],
            pin_memory=True,
        ),
    }

    return dataloaders, datasets["train"].class_to_idx
