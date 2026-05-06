"""Preprocessing pipeline orchestration (detect/align/rotate over a directory)."""

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm

from face_bias.preprocessing.detection import detect_and_adjust_faces, rotate_procedure


def detect_adjust_image(image_path, output_dir, face_detector, config: dict[str, Any]):
    """Detect faces, align and resize, then save each face as JPEG."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            logging.warning(f"Failed to load image: {image_path}")
            return

        output_size = tuple(config["image"]["image_size"])
        processed_faces = detect_and_adjust_faces(image, face_detector, output_size)

        for i, face in enumerate(processed_faces):
            output_path = output_dir / f"{Path(image_path).stem}_face_{i}.jpg"
            cv2.imwrite(str(output_path), face)
            logging.info(f"Processed face saved: {output_path}")

    except Exception as exc:  # noqa: BLE001 — defensive at OpenCV boundary
        logging.error(f"Error processing {image_path}: {exc}")


def rotate_image(image_path, output_dir, face_detector, config: dict[str, Any]):  # noqa: ARG001
    """Rotate an image in configured angle increments and save the results."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            logging.warning(f"Failed to load image: {image_path}")
            return

        for i in range(config["preprocessing"]["num_rotations"]):
            angle = i * config["preprocessing"]["rotate_angle"]
            rotated_image = rotate_procedure(image, angle)
            output_path = output_dir / f"{Path(image_path).stem}_rotated_{angle}.jpg"
            cv2.imwrite(str(output_path), rotated_image)
            logging.info(f"Rotated image saved: {output_path}")

    except Exception as exc:  # noqa: BLE001 — defensive at OpenCV boundary
        logging.error(f"Error rotating {image_path}: {exc}")


def process_preprocessing(input_dir, output_dir, face_detector, config: dict[str, Any]):
    """Run the configured preprocessing function over all images in input_dir."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    processing_functions = {
        "rotate_image": rotate_image,
        "detect_adjust_image": detect_adjust_image,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Starting preprocessing for input_dir: {input_dir}")

    image_paths = list(input_dir.rglob("*.jpg"))
    logging.info(f"Found {len(image_paths)} images in {input_dir}")

    processing_type = config["preprocessing"]["processing_type"]
    num_workers = config["preprocessing"]["num_workers"]
    fn = processing_functions[processing_type]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(fn, str(image_path), output_dir, face_detector, config)
            for image_path in image_paths
        ]
        for future in tqdm(futures, desc=f"Preprocessing ({processing_type})"):
            future.result()

    logging.info(f"Preprocessing finished. Output directory: {output_dir}")
