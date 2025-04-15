import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import cv2
import numpy as np

def detect_and_adjust_faces(image, face_detector, output_size):
    """
    Detect faces in the image and adjust them (crop, align, etc.).
    """
    try:
        # Detect faces and landmarks using MTCNN
        boxes, probs, landmarks = face_detector.detect(image, landmarks=True)
        processed_faces = []

        if boxes is None:
            logging.warning("No faces detected in the image.")
            return []

        for bbox, landmark in zip(boxes, landmarks):
            if bbox is None or landmark is None:
                continue

            # Crop the face
            cropped_face = cropping_procedure(image, bbox.astype(int))
            # Align the face using landmarks
            aligned_face = alignment_procedure(cropped_face, {
                'left_eye': landmark[0],
                'right_eye': landmark[1]
            })
            # Resize the face to the desired output size
            resized_face = cv2.resize(aligned_face, output_size)
            processed_faces.append(resized_face)

        return processed_faces
    except Exception as e:
        logging.error(f"Error detecting and adjusting faces: {e}")
        return []

def draw_bounding_procedure(image, bbox, color=(0, 255, 0), thickness=2):
    """
    Draw a bounding box on the image.
    """
    x, y, w, h = bbox
    return cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness)

def cropping_procedure(image, bbox, border=50):
    """
    Crop the given face in the image based on the provided bounding box.
    """
    x, y, w, h = bbox
    x1 = max(0, x - border)
    y1 = max(0, y - border)
    x2 = min(image.shape[1], x + w + border)
    y2 = min(image.shape[0], y + h + border)
    return image[y1:y2, x1:x2]

def rotate_procedure(image, angle):
    """
    Rotate the image by the specified angle.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix_rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix_rotation, (w, h))

def alignment_procedure(img, landmarks):
    """
    Align the image based on facial landmarks.
    """
    if not landmarks or len(landmarks) < 2:
        logging.warning("Insufficient landmarks for alignment")

    left_eye_x, left_eye_y = landmarks['left_eye']
    right_eye_x, right_eye_y = landmarks['right_eye']
    delta_x = right_eye_x - left_eye_x
    delta_y = right_eye_y - left_eye_y
    angle = np.degrees(np.arctan2(delta_y, delta_x))
    eyes_center = ((left_eye_x + right_eye_x) // 2, (left_eye_y + right_eye_y) // 2)
    matrix_rotation = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)

    return cv2.warpAffine(img, matrix_rotation, (img.shape[1], img.shape[0]))