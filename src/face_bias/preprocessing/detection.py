"""Face detection, cropping, alignment and rotation primitives."""

import logging

import cv2
import numpy as np


def detect_and_adjust_faces(image, face_detector, output_size):
    """Detect faces in the image and adjust them (crop, align, resize)."""
    try:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, _probs, landmarks = face_detector.detect(img_rgb, landmarks=True)
        processed_faces = []

        if boxes is None:
            logging.warning("No faces detected in the image.")
            return []

        for bbox, landmark in zip(boxes, landmarks):
            if bbox is None or landmark is None:
                continue

            cropped_face = cropping_procedure(img_rgb, bbox.astype(int))
            if cropped_face is None or cropped_face.size == 0:
                logging.warning("Failed to crop face.")
                continue

            aligned_face = alignment_procedure(
                cropped_face,
                {"left_eye": landmark[0], "right_eye": landmark[1]},
            )
            if aligned_face is None or aligned_face.size == 0:
                logging.warning("Failed to align face.")
                continue

            resized_face = cv2.resize(aligned_face, output_size, interpolation=cv2.INTER_AREA)
            color_adjusted_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
            processed_faces.append(color_adjusted_face)

        return processed_faces
    except Exception as exc:  # noqa: BLE001 — defensive at OpenCV/MTCNN boundary
        logging.error(f"Error detecting and adjusting faces: {exc}")
        return []


def cropping_procedure(img, bbox, border=70):
    """Crop the given face in img based on given coordinates."""
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    return img[y1 - border : y1 + height + border, x1 - border : x1 + width + border]


def rotate_procedure(image, angle):
    """Rotate the image by the specified angle."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix_rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix_rotation, (w, h))


def alignment_procedure(img, landmarks):
    """Align the image based on left/right eye landmarks."""
    if not landmarks or len(landmarks) < 2:
        logging.warning("Insufficient landmarks for alignment")

    left_eye_x, left_eye_y = landmarks["left_eye"]
    right_eye_x, right_eye_y = landmarks["right_eye"]
    delta_x = right_eye_x - left_eye_x
    delta_y = right_eye_y - left_eye_y
    angle = np.degrees(np.arctan2(delta_y, delta_x))
    eyes_center = ((left_eye_x + right_eye_x) // 2, (left_eye_y + right_eye_y) // 2)
    matrix_rotation = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)

    return cv2.warpAffine(img, matrix_rotation, (img.shape[1], img.shape[0]))
