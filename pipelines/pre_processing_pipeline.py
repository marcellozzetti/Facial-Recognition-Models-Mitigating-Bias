import os
import cv2
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from preprocessing.pre_processing_images import detect_and_adjust_faces, draw_bounding_procedure
from utils.config import load_config
from utils.custom_logging import setup_logging

def process_image(image_path, output_dir, face_detector, config):
    """
    Process a single image through the pipeline.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            logging.warning(f"Failed to load image: {image_path}")
            return

        output_size = tuple(config['output_size'])
        processed_faces = detect_and_adjust_faces(image, face_detector, output_size)

        for i, face in enumerate(processed_faces):
            output_path = output_dir / f"{Path(image_path).stem}_face_{i}.jpg"
            cv2.imwrite(str(output_path), face)
            logging.info(f"Processed face saved: {output_path}")

    except Exception as e:
        logging.error(f"Error processing {image_path}: {e}")

def process_directory(input_dir, output_dir, face_detector, config):
    """
    Process all images in a directory through the pipeline.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(input_dir.rglob('*.jpg'))
    logging.info(f"Found {len(image_paths)} images in {input_dir}")

    with ThreadPoolExecutor(max_workers=config.get('num_workers', 4)) as executor:
        futures = [
            executor.submit(process_image, str(image_path), output_dir, face_detector, config)
            for image_path in image_paths
        ]
        for future in tqdm(futures, desc="Processing images"):
            future.result()

def main():
    config = load_config('configs/default.yaml')
    setup_logging(config, 'log_preprocessing_file')

    # Initialize face detector (replace with actual detector)
    face_detector = MTCNN()

    process_directory(
        config['preprocessing']['input_dir'],
        config['preprocessing']['output_dir'],
        face_detector,
        config['preprocessing']
    )

if __name__ == "__main__":
    main()