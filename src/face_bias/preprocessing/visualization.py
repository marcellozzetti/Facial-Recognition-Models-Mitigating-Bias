"""Visualisation helpers (bounding boxes, landmarks)."""

import logging

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


def draw_bounding_procedure(img, bbox, landmark):
    """Draw bounding box and 5-point landmarks on an image using matplotlib."""
    try:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax = plt.gca()

        x1, y1, x2, y2 = bbox.astype(int)
        width, height = x2 - x1, y2 - y1
        rect = Rectangle((x1, y1), width, height, fill=False, color="red", linewidth=2)
        ax.add_patch(rect)

        left_eye, right_eye, nose, mouth_left, mouth_right = landmark
        ax.add_patch(Circle(left_eye, radius=2, color="blue"))
        ax.add_patch(Circle(right_eye, radius=2, color="blue"))
        ax.add_patch(Circle(nose, radius=2, color="green"))
        ax.add_patch(Circle(mouth_left, radius=2, color="orange"))
        ax.add_patch(Circle(mouth_right, radius=2, color="orange"))

        ax.text(left_eye[0], left_eye[1], "Left Eye", fontsize=8, color="blue", verticalalignment="bottom")
        ax.text(right_eye[0], right_eye[1], "Right Eye", fontsize=8, color="blue", verticalalignment="bottom")
        ax.text(nose[0], nose[1], "Nose", fontsize=8, color="green", verticalalignment="bottom")
        ax.text(mouth_left[0], mouth_left[1], "Mouth Left", fontsize=8, color="orange", verticalalignment="bottom")
        ax.text(mouth_right[0], mouth_right[1], "Mouth Right", fontsize=8, color="orange", verticalalignment="bottom")

        plt.axis("off")
        plt.show()

    except Exception as exc:  # noqa: BLE001 — defensive at matplotlib boundary
        logging.error(f"Error drawing bounding box and keypoints: {exc}")
