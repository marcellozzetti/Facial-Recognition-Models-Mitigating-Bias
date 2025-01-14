import os
import cv2
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from utils.config import load_config
from utils.logging import setup_logging
from preprocessing.transformations import cropping_procedure, rotate_procedure, alignment_procedure

def draw_bounding_procedure(img):
    """
    Detect faces and draw bounding boxes.
    """
    plt.imshow(img)
    ax = plt.gca()
    detector = MTCNN()
    detect_faces = detector.detect_faces(img)
    if len(detect_faces) == 0:
        print("No face detected.")
        return
    keypoints = detect_faces[0]['keypoints']
    x, y, width, height = detect_faces[0]['box']
    rect = Rectangle((x, y), width, height, fill=False, color='red')
    ax.add_patch(rect)
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']
    nose = keypoints['nose']
    mouth_left = keypoints['mouth_left']
    mouth_right = keypoints['mouth_right']
    ax.add_patch(Circle(left_eye, radius=2, color='blue'))
    ax.add_patch(Circle(right_eye, radius=2, color='blue'))
    ax.add_patch(Circle(nose, radius=2, color='green'))
    ax.add_patch(Circle(mouth_left, radius=2, color='orange'))
    ax.add_patch(Circle(mouth_right, radius=2, color='orange'))
    ax.text(left_eye[0], left_eye[1], 'Left Eye', fontsize=8, color='blue', verticalalignment='bottom')
    ax.text(right_eye[0], right_eye[1], 'Right Eye', fontsize=8, color='blue', verticalalignment='bottom')
    ax.text(nose[0], nose[1], 'Nose', fontsize=8, color='green', verticalalignment='bottom')
    ax.text(mouth_left[0], mouth_left[1], 'Mouth Left', fontsize=8, color='orange', verticalalignment='bottom')
    ax.text(mouth_right[0], mouth_right[1], 'Mouth Right', fontsize=8, color='orange', verticalalignment='bottom')
    plt.axis('off')
    plt.show()

def detect_and_adjust_faces(img, img_name, save_dir=None, draw_bounding=False):
    """
    Detect and perform face adjustments.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    detect_faces = detector.detect_faces(img_rgb)
    if len(detect_faces) == 0:
        print("No face detected: {}".format(img_name))
        return False
    keypoints = detect_faces[0]['keypoints']
    x, y, width, height = detect_faces[0]['box']
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']
    aligned_face = alignment_procedure(img_rgb, left_eye, right_eye)
    treated_face = cropping_procedure(aligned_face, x, y, width, height)
    if draw_bounding:
        draw_bounding_procedure(treated_face)
    try:
        treated_face = cv2.resize(treated_face, (224, 224), interpolation=cv2.INTER_AREA)
    except cv2.error as e:
        print(f"Failed to resize image {img_name}. Skipping.")
        return False
    treated_face = cv2.cvtColor(treated_face, cv2.COLOR_BGR2RGB)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, img_name)
        print("Save treated image:", save_path)
        cv2.imwrite(save_path, treated_face)
    return True


def process_image(image_path, output_path, image_size):
    """
    Process a single image: read, resize, and save.
    """
    try:
        # Read the image
        image = cv2.imread(str(image_path))
        if image is None:
            logging.warning(f"Failed to read image: {image_path}")
            return

        # Resize the image
        resized_image = cv2.resize(image, image_size)

        # Ensure the output directory exists
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the processed image
        cv2.imwrite(str(output_path), resized_image)
        logging.info(f"Processed and saved image: {output_path}")
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")

def process_images_in_directory(input_dir, output_dir, image_size, num_workers):
    """
    Process all images in the input directory and save them to the output directory.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Gather all image paths
    image_paths = list(input_dir.rglob('*.[jp][pn]g'))  # Matches .jpg, .jpeg, .png
    logging.info(f"Found {len(image_paths)} images in {input_dir}")

    # Process images in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                process_image,
                image_path,
                output_dir / image_path.relative_to(input_dir),
                image_size
            )
            for image_path in image_paths
        ]
        for future in tqdm(futures, desc="Processing images"):
            future.result()  # Wait for all futures to complete

def main():
    # Load configuration
    config = load_config('configs/default.yaml')

    # Set up logging
    setup_logging(config, 'preprocessing_log_file')

    # Process images
    process_images_in_directory(
        input_dir=config['preprocessing']['input_dir'],
        output_dir=config['preprocessing']['output_dir'],
        image_size=tuple(config['preprocessing']['image_size']),
        num_workers=config['preprocessing'].get('num_workers', 4)
    )

if __name__ == "__main__":
    main()