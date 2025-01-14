import cv2
import logging
import numpy as np

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