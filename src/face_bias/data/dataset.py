"""PyTorch Dataset and DataLoader builders for FairFace."""

import logging
import os
from typing import Any

import cv2
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class FaceDataset(Dataset):
    """Dataset that yields (CHW float tensor, encoded label) pairs."""

    def __init__(self, img_paths, labels, img_dir, transform=None, label_encoder=None):
        if label_encoder is None:
            raise ValueError("label_encoder is required so labels can be transformed.")
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

        # cv2 returns BGR uint8 ndarray; convert to RGB and wrap in PIL so
        # torchvision transforms (Resize, RandomHorizontalFlip, ToTensor) work
        # on a PIL.Image as they expect.
        img_bgr = cv2.imread(img_name)
        if img_bgr is None:
            raise FileNotFoundError(f"Image {img_name} not found or unreadable")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)

        if self.transform:
            img = self.transform(img)

        label_index = int(self.label_encoder.transform([label])[0])
        return img, label_index


def _build_transforms(config: dict[str, Any]) -> dict[str, transforms.Compose]:
    image_size = tuple(config["image"]["image_size"])
    image_mean = config["image"]["image_mean"]
    image_std = config["image"]["image_std"]

    eval_pipeline = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(image_mean, image_std),
        ]
    )
    train_pipeline = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(image_mean, image_std),
        ]
    )

    return {"train": train_pipeline, "val": eval_pipeline, "test": eval_pipeline}


def setup_dataset(config: dict[str, Any]):
    """Build train/val/test DataLoaders.

    Returns:
        dataloaders: dict with keys 'train', 'val', 'test'
        label_encoder: fitted sklearn LabelEncoder (use .classes_ to recover names)
        num_classes: int
    """
    data_transforms = _build_transforms(config)

    csv_pd = pd.read_csv(config["data"]["dataset_file"])

    label_encoder = LabelEncoder()
    label_encoder.fit(csv_pd["race"])
    num_classes = len(label_encoder.classes_)
    logging.info(f"{num_classes} classes: {list(label_encoder.classes_)}")

    X = csv_pd["file"]
    y = csv_pd["race"]

    test_size = config["training"]["test_size"]
    random_state = config["training"]["random_state"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=test_size,
        stratify=y_train,
        random_state=random_state,
    )

    image_dir = config["data"]["dataset_image_output_path"]

    datasets = {
        split: FaceDataset(
            X_split.tolist(),
            y_split.tolist(),
            image_dir,
            transform=data_transforms[split],
            label_encoder=label_encoder,
        )
        for split, (X_split, y_split) in {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }.items()
    }

    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]

    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    return dataloaders, label_encoder, num_classes
