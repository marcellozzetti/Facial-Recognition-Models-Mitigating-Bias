import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from facenet_pytorch import MTCNN
from preprocessing.pre_processing_images import detect_and_adjust_faces, rotate_procedure
from utils.config import load_config
from utils.custom_logging import setup_logging

def detect_adjust_image(image_path, output_dir, face_detector, config):
    """
    Process a single image through the pipeline.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            logging.warning(f"Failed to load image: {image_path}")
            return

        output_size = tuple(config['image']['image_size'])
        processed_faces = detect_and_adjust_faces(image, face_detector, output_size)

        for i, face in enumerate(processed_faces):
            output_path = output_dir / f"{Path(image_path).stem}_face_{i}.jpg"
            cv2.imwrite(str(output_path), face)
            logging.info(f"Processed face saved: {output_path}")

    except Exception as e:
        logging.error(f"Error processing {image_path}: {e}")

def rotate_image(image_path, output_dir, face_detector, config):
    """
    Rotate an image in a configured degree increments and save the results.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            logging.warning(f"Failed to load image: {image_path}")
            return

        for i in range(config['preprocessing']['num_rotations']):

            angle = i * config['preprocessing']['rotate_angle']
            # Perform the rotation
            rotated_image = rotate_procedure(image, angle)
            # Save the rotated image
            output_path = output_dir / f"{Path(image_path).stem}_rotated_{angle}.jpg"
            cv2.imwrite(str(output_path), rotated_image)
            logging.info(f"Rotated image saved: {output_path}")

    except Exception as e:
        logging.error(f"Error rotating {image_path}: {e}")

def process_preprocessing(input_dir, output_dir, face_detector, config):
    """
    Process the pre processing over all images in a directory through the pipeline.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    PROCESSING_FUNCTIONS = {
        "rotate_image": rotate_image,
        "detect_adjust_image": detect_adjust_image
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Starting process directory for input_dir: {input_dir}")

    image_paths = list(input_dir.rglob('*.jpg'))
    logging.info(f"Found {len(image_paths)} images in {input_dir}")

    with ThreadPoolExecutor(max_workers=config['preprocessing']['num_workers']) as executor:
        futures = [
            executor.submit(
                PROCESSING_FUNCTIONS[config['preprocessing']['processing_type']],
                str(image_path),
                output_dir,
                face_detector,
                config
            )
            for image_path in image_paths
        ]
        for future in tqdm(futures, desc="Processing images"):
            logging.info(f"Processing images for processing_type: {config['preprocessing']['processing_type']}")
            future.result()
    
    logging.info(f"Ending process directory for output_dir: {output_dir}")

def main():
    config = load_config('configs/default.yaml')
    setup_logging(config, 'log_preprocessing_file')

    logging.info(f"Starting preprocessing with config: {config}")

    # Initialize face detector
    face_detector = MTCNN()

    process_preprocessing(
        config['data']['dataset_image_input_path'],
        config['data']['dataset_image_output_path'],
        face_detector,
        config
    )

    logging.info(f"Ending preprocessing")

if __name__ == "__main__":
    main()