import os
import cv2
import logging
from pathlib import Path
from preprocessing.pre_processing_images import detect_and_adjust_faces, draw_bounding_procedure
from utils.config import load_config
from utils.logging import setup_logging

def preprocess_image_pipeline(image_path, output_path, config):
    """
    Full preprocessing pipeline for a single image.
    Steps:
        1. Load the image.
        2. Detect faces.
        3. Draw bounding boxes (optional for visualization).
        4. Crop detected faces.
        5. Align faces.
        6. Resize faces.
        7. Save preprocessed faces.
        8. Log success or failure.
    """
    logging.info(f"Starting pipeline for image: {image_path}")

    # Step 1: Load the image
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Failed to load image: {image_path}")
        return

    # Step 2: Detect faces
    face_detector = config['face_detector']  # Replace with actual face detector object
    output_size = tuple(config['output_size'])
    faces = detect_and_adjust_faces(image, face_detector, output_size)

    # Step 3: Optional - Draw bounding boxes
    for face in faces:
        bbox = face.get('bbox')
        image = draw_bounding_procedure(image, bbox)

    # Step 4 to 6: Process detected faces
    processed_faces = []
    for face in faces:
        cropped_face = face.get('cropped')
        aligned_face = face.get('aligned')
        resized_face = cv2.resize(aligned_face, output_size)
        processed_faces.append(resized_face)

    # Step 7: Save processed faces
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, face in enumerate(processed_faces):
        face_path = output_dir / f"{Path(image_path).stem}_face_{i}.jpg"
        cv2.imwrite(str(face_path), face)
        logging.info(f"Processed face saved: {face_path}")

    # Step 8: Log success
    logging.info(f"Pipeline completed successfully for image: {image_path}")

def main():
    # Load configuration
    config = load_config('configs/default.yaml')

    # Set up logging
    setup_logging(config, 'preprocessing_log_file')

    # Define input and output directories
    input_dir = Path(config['preprocessing']['input_dir'])
    output_dir = Path(config['preprocessing']['output_dir'])

    # Process each image
    for image_path in input_dir.rglob('*.jpg'):
        output_path = output_dir / image_path.relative_to(input_dir)
        preprocess_image_pipeline(str(image_path), str(output_path), config)

if __name__ == "__main__":
    main()