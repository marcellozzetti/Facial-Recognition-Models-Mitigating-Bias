"""PyTorch Dataset and DataLoader builders for FairFace."""

import logging
import os
from typing import Any

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class FaceDataset(Dataset):
    """Dataset that yields (CHW float tensor, encoded int label) pairs.

    Labels are pre-encoded once in ``setup_dataset`` and passed in as a
    list of ints — calling ``LabelEncoder.transform`` per ``__getitem__``
    was a significant overhead with 4 workers under Windows.

    Image decoding uses ``PIL.Image.open(...).convert("RGB")`` directly
    rather than ``cv2.imread`` + ``cv2.cvtColor`` + ``PIL.fromarray``:
    one syscall, one decode, no intermediate numpy array.
    """

    def __init__(self, img_paths, encoded_labels, img_dir, transform=None):
        if len(img_paths) != len(encoded_labels):
            raise ValueError("img_paths and encoded_labels must have the same length")
        self.img_paths = img_paths
        self.encoded_labels = encoded_labels
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_paths[idx])
        with Image.open(img_name) as raw:
            img = raw.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(self.encoded_labels[idx])


def _build_transforms(config: dict[str, Any]) -> dict[str, transforms.Compose]:
    image_size = tuple(config["image"]["image_size"])
    image_mean = config["image"]["image_mean"]
    image_std = config["image"]["image_std"]
    # Optional FineFACE-recipe-style aug (RandomCrop with padding after
    # Resize). None preserves the prior project-wide behaviour: HFlip only.
    rcrop_pad = config["training"].get("train_random_crop_padding")

    eval_pipeline = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(image_mean, image_std),
        ]
    )
    train_ops: list = [transforms.Resize(image_size)]
    if rcrop_pad:
        train_ops.append(
            transforms.RandomCrop(image_size, padding=int(rcrop_pad))
        )
    train_ops += [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(image_mean, image_std),
    ]
    train_pipeline = transforms.Compose(train_ops)

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

    Reproduces the MBA Cap. 4 "Sample count after balancing" table: each
    ``race`` class is downsampled to the size of the smallest class.
    Returns a new dataframe in shuffled order.
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
    val_size = config["training"].get("val_size")
    random_state = config["training"]["random_state"]

    # Stage 1: peel off the test set as `test_size` of the whole dataset.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Stage 2: peel off the validation set from the remaining train+val pool.
    # When val_size is set, it is the *final* validation fraction of the whole
    # dataset (so test_size=0.1 + val_size=0.1 -> 80/10/10). When omitted,
    # we keep the legacy MBA cascade: train_test_split called twice with the
    # same `test_size`, which yields ~64/16/20 for test_size=0.2.
    if val_size is None:
        stage2_test_size = test_size
        logging.info(
            f"Split (legacy cascade): test={test_size:.0%}, "
            f"val~{test_size * (1 - test_size):.0%}, "
            f"train~{(1 - test_size) ** 2:.0%}"
        )
    else:
        stage2_test_size = val_size / (1 - test_size)
        logging.info(
            f"Split: train={1 - val_size - test_size:.0%}, val={val_size:.0%}, test={test_size:.0%}"
        )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=stage2_test_size,
        stratify=y_train_val,
        random_state=random_state,
    )

    # Encode the string labels once up-front so __getitem__ doesn't have
    # to call LabelEncoder.transform per sample in every worker.
    encoded = {
        "train": label_encoder.transform(y_train).tolist(),
        "val": label_encoder.transform(y_val).tolist(),
        "test": label_encoder.transform(y_test).tolist(),
    }

    splits = {"train": (X_train, encoded["train"]), "val": (X_val, encoded["val"]), "test": (X_test, encoded["test"])}
    datasets = {
        split: FaceDataset(
            X_split.tolist(),
            encoded_labels,
            image_dir,
            transform=data_transforms[split],
        )
        for split, (X_split, encoded_labels) in splits.items()
    }

    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]
    prefetch_factor = config["training"].get("prefetch_factor", 4)
    # persistent_workers requires num_workers > 0, and saves the per-epoch
    # worker re-spawn cost (3-5 s on Windows). DataLoader rejects the
    # combination silently in some versions, so guard explicitly.
    persistent = num_workers > 0

    def _loader(ds, *, shuffle: bool) -> DataLoader:
        kwargs: dict[str, Any] = dict(
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
        if num_workers > 0:
            kwargs["persistent_workers"] = persistent
            kwargs["prefetch_factor"] = prefetch_factor
        return DataLoader(ds, **kwargs)

    dataloaders = {
        "train": _loader(datasets["train"], shuffle=True),
        "val": _loader(datasets["val"], shuffle=False),
        "test": _loader(datasets["test"], shuffle=False),
    }

    return dataloaders, label_encoder, num_classes
