import logging
import cv2
from transformations import cropping_procedure, alignment_procedure

def detect_and_adjust_faces(image, face_detector, output_size):
    """
    Detect faces in the image and adjust them (crop, align, etc.).
    """
    try:
        faces = face_detector.detect_faces(image)
        processed_faces = []

        for face in faces:
            bbox = face['bbox']
            cropped_face = cropping_procedure(image, bbox)
            aligned_face = alignment_procedure(cropped_face, face.get('landmarks', []))
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