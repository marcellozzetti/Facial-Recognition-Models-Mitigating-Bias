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


def _filter_existing_images(csv_pd: pd.DataFrame, image_dir: str) -> pd.DataFrame:
    """Drop rows whose image file is missing on disk.

    Some FairFace photos failed MTCNN detection during preprocessing and
    therefore have no aligned counterpart on disk. The trainer/evaluator
    would crash mid-epoch when a worker tried to read them; filter once
    upfront instead.
    """
    paths = csv_pd["file"].apply(lambda f: os.path.join(image_dir, f))
    exists_mask = paths.apply(os.path.isfile)
    missing = (~exists_mask).sum()
    if missing:
        logging.warning(f"Dropping {missing} rows whose image file is missing under {image_dir}")
    return csv_pd.loc[exists_mask].reset_index(drop=True)


def _filter_by_class(
    csv_pd: pd.DataFrame, label_column: str, allowed: list[str] | None
) -> pd.DataFrame:
    """Restrict the dataframe to rows whose label is in ``allowed``.

    Reproduces the MBA's Exp. 7/8 setup ("classes alvo: White e Black").
    Returns the dataframe unchanged when ``allowed`` is None or empty.
    """
    if not allowed:
        return csv_pd
    before = len(csv_pd)
    out = csv_pd.loc[csv_pd[label_column].isin(allowed)].reset_index(drop=True)
    logging.info(f"Class filter {sorted(allowed)}: kept {len(out)}/{before} rows")
    return out


def _undersample_to_minority(
    csv_pd: pd.DataFrame, label_column: str, random_state: int
) -> pd.DataFrame:
    """Random undersample so every class has the same count as the minority.

    Reproduces the MBA's ``Contagem de Amostras Após o Balanceamento`` from
    Cap. 4: each ``race`` class is downsampled to the size of the smallest
    class. Returns a new dataframe in shuffled order.
    """
    counts = csv_pd[label_column].value_counts()
    minority = int(counts.min())
    logging.info(
        f"Undersampling {label_column} to minority size {minority} (was: {counts.to_dict()})"
    )
    parts = [
        group.sample(n=minority, random_state=random_state)
        for _, group in csv_pd.groupby(label_column, sort=False)
    ]
    out = pd.concat(parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    logging.info(f"Balanced dataset size: {len(out)} ({len(counts)} classes x {minority})")
    return out


def setup_dataset(config: dict[str, Any]):
    """Build train/val/test DataLoaders.

    Returns:
        dataloaders: dict with keys 'train', 'val', 'test'
        label_encoder: fitted sklearn LabelEncoder (use .classes_ to recover names)
        num_classes: int
    """
    data_transforms = _build_transforms(config)

    image_dir = config["data"]["dataset_image_output_path"]
    csv_pd = pd.read_csv(config["data"]["dataset_file"])
    csv_pd = _filter_existing_images(csv_pd, image_dir)
    csv_pd = _filter_by_class(csv_pd, "race", config["data"].get("class_filter"))

    if config["data"].get("balance", "none") == "undersample":
        csv_pd = _undersample_to_minority(
            csv_pd,
            label_column="race",
            random_state=config["training"]["random_state"],
        )

    label_encoder = LabelEncoder()
    label_encoder.fit(csv_pd["race"])
    num_classes = len(label_encoder.classes_)
    logging.info(f"{num_classes} classes: {list(label_encoder.classes_)}")
    logging.info(f"{len(csv_pd)} images after filtering missing files")

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
