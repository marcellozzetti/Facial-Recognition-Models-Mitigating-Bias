import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

def detect_and_adjust_faces(image, face_detector, output_size):
    """
    Detect faces in the image and adjust them (crop, align, etc.).
    """
    try:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Detect faces and landmarks using MTCNN
        boxes, probs, landmarks = face_detector.detect(img_rgb, landmarks=True)
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
            
            # Align the face using landmarks
            aligned_face = alignment_procedure(cropped_face, {
                'left_eye': landmark[0],
                'right_eye': landmark[1]
            })
            if aligned_face is None or aligned_face.size == 0:
                logging.warning("Failed to align face.")
                continue

            # Resize the face to the desired output size
            resized_face = cv2.resize(aligned_face, output_size, interpolation=cv2.INTER_AREA)
            color_adjusted_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
            processed_faces.append(color_adjusted_face)

        return processed_faces
    except Exception as e:
        logging.error(f"Error detecting and adjusting faces: {e}")
        return []

def draw_bounding_procedure(img, bbox, landmark):
    """
    Detect faces and draw bounding boxes and keypoints on the image using matplotlib.
    """
    try:
        # Plot the image
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax = plt.gca()

        # Draw bounding box
        x1, y1, x2, y2 = bbox.astype(int)
        width, height = x2 - x1, y2 - y1
        rect = Rectangle((x1, y1), width, height, fill=False, color='red', linewidth=2)
        ax.add_patch(rect)

        # Draw keypoints
        left_eye, right_eye, nose, mouth_left, mouth_right = landmark
        ax.add_patch(Circle(left_eye, radius=2, color='blue'))
        ax.add_patch(Circle(right_eye, radius=2, color='blue'))
        ax.add_patch(Circle(nose, radius=2, color='green'))
        ax.add_patch(Circle(mouth_left, radius=2, color='orange'))
        ax.add_patch(Circle(mouth_right, radius=2, color='orange'))

        # Add labels for keypoints
        ax.text(left_eye[0], left_eye[1], 'Left Eye', fontsize=8, color='blue', verticalalignment='bottom')
        ax.text(right_eye[0], right_eye[1], 'Right Eye', fontsize=8, color='blue', verticalalignment='bottom')
        ax.text(nose[0], nose[1], 'Nose', fontsize=8, color='green', verticalalignment='bottom')
        ax.text(mouth_left[0], mouth_left[1], 'Mouth Left', fontsize=8, color='orange', verticalalignment='bottom')
        ax.text(mouth_right[0], mouth_right[1], 'Mouth Right', fontsize=8, color='orange', verticalalignment='bottom')

        plt.axis('off')
        plt.show()

    except Exception as e:
        logging.error(f"Error drawing bounding box and keypoints: {e}")

def cropping_procedure(img, bbox, border=70):
    """
    Crop the given face in img based on given coordinates.
    """
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1

    return img[y1-border:y1+height+border, x1-border:x1+width+border]

#def cropping_procedure(image, bbox, border=0):
    """
    Crop the given face in the image based on the bounding box coordinates.
    Ensures the cropping stays within the image boundaries.
    """
    try:
        # Extraia as coordenadas do bounding box
        x, y, w, h = bbox
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)

        # Adicione uma borda opcional ao redor do bounding box
        x1 = max(0, x1 - border)
        y1 = max(0, y1 - border)
        x2 = min(image.shape[1], x2 + border)
        y2 = min(image.shape[0], y2 + border)

        # Realize o corte
        cropped_image = image[y1:y2, x1:x2]

        if cropped_image.size == 0:
            logging.warning(f"Cropping resulted in an empty image. bbox: {bbox}")
            return None

        return cropped_image
    except Exception as e:
        logging.error(f"Error during cropping: {e}")
        return None

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