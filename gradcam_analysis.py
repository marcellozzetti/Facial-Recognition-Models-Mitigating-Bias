import os
import torch
import torch.nn as nn
from torchcam.methods import GradCAM
import cv2
import numpy as np

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
TARGET_LAYER_NAME = 'backbone.layer4.2.conv3'
DEFAULT_SAVE_DIR = 'output/grad_cam'

# Available colormaps
COLORMAPS = {
    "jet": cv2.COLORMAP_JET
}

def get_target_layer(model: nn.Module, layer_name: str):
    """
    Retrieves the target layer from the model.
    """
    target_layer = dict(model.named_modules()).get(layer_name)
    if target_layer is None:
        raise ValueError(f"Target layer '{layer_name}' not found in the model.")
    return target_layer

def denormalize_image(tensor, mean, std):
    """
    Desnormaliza a imagem para o intervalo [0, 255].
    """
    tensor = tensor.clone().detach()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def save_grad_cam_visualization(image: np.ndarray, cam_image: np.ndarray, 
                                save_path: str, colormap: int, original_save_path: str):
    """
    Saves Grad-CAM visualization and the original image.
    """
    # Ensure that the Grad-CAM image has the same size as the original image
    cam_image_resized = cv2.resize(cam_image, (image.shape[1], image.shape[0]))

    # Convert the cam_image to a 3-channel image if it's not already (it could be grayscale)
    if len(cam_image_resized.shape) == 2:  # Grayscale heatmap, convert to 3 channels
        cam_image_resized = cv2.cvtColor(cam_image_resized, cv2.COLOR_GRAY2BGR)

    # Apply colormap to the heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_image_resized), colormap)

    # Ensure both image and heatmap are of type uint8
    image = np.uint8(image)

    # Superimpose the original image with the heatmap
    superimposed_image = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    # Save the Grad-CAM visualization
    cv2.imwrite(save_path, superimposed_image)
    print(f"Grad-CAM saved: {save_path}")

    # Save the original image
    cv2.imwrite(original_save_path, image)
    print(f"Original image saved: {original_save_path}")

def generate_grad_cam(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, 
                      incorrect_indices: list, label_encoder=None, save_dir: str = DEFAULT_SAVE_DIR):
                          
    """
    Generates Grad-CAM visualizations for incorrectly classified images.
    """
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # Handle DataParallel wrapper

    # Initialize Grad-CAM extractor
    cam_extractor = GradCAM(model, target_layer=get_target_layer(model, TARGET_LAYER_NAME))

    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    CLASS_NAMES = label_encoder.classes_.tolist()  # Get class names from the encoder
    print("Classes: ", CLASS_NAMES)

    for idx in incorrect_indices:
        image = images[idx].unsqueeze(0).to(DEVICE)
        label = labels[idx].item()

        # Forward pass and get predictions
        output = model(image)
        pred_class = output.squeeze(0).argmax().item()

        # Ensure prediction index is valid
        if pred_class >= output.size(1):
            raise ValueError(f"Predicted class index {pred_class} is out of bounds.")

        # Generate Grad-CAM mask
        cam_image = cam_extractor(class_idx=pred_class, scores=output)
        cam_image = cam_image[0] if isinstance(cam_image, list) else cam_image
        cam_image = cam_image.squeeze().cpu().numpy()
        cam_image = np.maximum(cam_image, 0) / cam_image.max()
        cam_image = cv2.resize(cam_image, (image.shape[3], image.shape[2]))

        # Convert the original image to NumPy
        original_image = image.squeeze().cpu().numpy().transpose((1, 2, 0))

        # Denormalizar a imagem
        original_image = denormalize_image(torch.tensor(original_image), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        original_image = np.uint8(original_image.permute(1, 2, 0).cpu().numpy() * 255)  # De volta para o intervalo [0, 255]

        # Save Grad-CAM visualizations with different colormaps
        for cmap_name, cmap in COLORMAPS.items():
            output_filename = os.path.join(
                save_dir, f'grad_cam_true_{CLASS_NAMES[label]}_pred_{CLASS_NAMES[pred_class]}_{idx}_{cmap_name}.png'
            )
            original_filename = os.path.join(
                save_dir, f'original_true_{CLASS_NAMES[label]}_pred_{CLASS_NAMES[pred_class]}_{idx}.png'
            )
            save_grad_cam_visualization(original_image, cam_image, output_filename, cmap, original_filename)

def main():
    """
    Main function to execute Grad-CAM generation.
    """
    from models import LResNet50E_IR  # Placeholder for your model import

    # Initialize the model
    num_classes = 7
    model = LResNet50E_IR(num_classes=num_classes).to(DEVICE)
    model.eval()

    # Example input tensor and labels for testing
    images = torch.randn(10, 3, 224, 224).to(DEVICE)  # Random example images
    labels = torch.randint(0, num_classes, (10,)).to(DEVICE)  # Random labels
    incorrect_indices = [i for i in range(10)]  # Example: process all images

    # Generate Grad-CAM for the incorrect samples
    generate_grad_cam(model, images, labels, incorrect_indices)

if __name__ == "__main__":
    main()
