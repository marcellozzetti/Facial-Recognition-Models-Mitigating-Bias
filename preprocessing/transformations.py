import cv2
import numpy as np

def cropping_procedure(img, x, y, width, height):
    """
    Crop the given face in img based on given coordinates.
    """
    border = 60
    return img[y-border:y+height+border, x-border:x+width+border]

def rotate_procedure(img, angle):
    """
    Rotate the image by the specified angle.
    """
    center = (img.shape[1] // 2, img.shape[0] // 2)
    matrix_rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, matrix_rotation, (img.shape[1], img.shape[0]))

def alignment_procedure(img, left_eye, right_eye):
    """
    Align the image based on facial landmarks.
    """
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye
    delta_x = right_eye_x - left_eye_x
    delta_y = right_eye_y - left_eye_y
    angle = np.degrees(np.arctan2(delta_y, delta_x))
    eyes_center = ((left_eye_x + right_eye_x) // 2, (left_eye_y + right_eye_y) // 2)
    matrix_rotation = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)

    return cv2.warpAffine(img, matrix_rotation, (img.shape[1], img.shape[0]))