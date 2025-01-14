import cv2
from preprocessing.pre_processing_images import draw_bounding_procedure

def test_draw_faces():
    """
    Test the draw_bounding_procedure function by drawing bounding boxes on an example image.
    """
    # Step 1: Load an example image
    input_image_path = "path/to/example.jpg"  # Replace with a valid path
    image = cv2.imread(input_image_path)

    if image is None:
        print("Failed to load image. Please check the path.")
        return

    # Step 2: Define bounding boxes (x, y, width, height)
    bounding_boxes = [
        (50, 50, 100, 100),  # Example box 1
        (200, 150, 120, 120)  # Example box 2
    ]

    # Step 3: Draw bounding boxes on the image
    for bbox in bounding_boxes:
        image = draw_bounding_procedure(image, bbox)

    # Step 4: Save or display the result
    output_image_path = "output_with_faces.jpg"
    cv2.imwrite(output_image_path, image)
    print(f"Processed image saved at: {output_image_path}")

    # Optional: Display the image
    cv2.imshow("Image with Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_draw_faces()