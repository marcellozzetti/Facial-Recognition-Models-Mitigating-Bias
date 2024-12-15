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
    #"hot": cv2.COLORMAP_HOT,
    #"cool": cv2.COLORMAP_COOL,
    #"rainbow": cv2.COLORMAP_RAINBOW
}

def get_target_layer(model: nn.Module, layer_name: str):
    """
    Retrieves the target layer from the model.
    Args:
        model (nn.Module): The neural network model.
        layer_name (str): The name of the target layer.
    Returns:
        nn.Module: The target layer module.
    Raises:
        ValueError: If the target layer is not found.
    """
    target_layer = dict(model.named_modules()).get(layer_name)
    if target_layer is None:
        raise ValueError(f"Target layer '{layer_name}' not found in the model.")
    return target_layer

def list_model_layers(model: nn.Module):
    """
    Lists all the layers in the model with their names.
    Args:
        model (nn.Module): The neural network model.
    """
    print("Listing model layers:")
    for name, module in model.named_modules():
        print(name)

def register_layer_hook(layer: nn.Module):
    """
    Registers a forward hook to capture activations of a given layer.
    Args:
        layer (nn.Module): Target layer for activation capture.
    Returns:
        Tuple: (hook, activations getter function)
    """
    activations = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    hook = layer.register_forward_hook(forward_hook)
    return hook, lambda: activations

def save_grad_cam_visualization(image: np.ndarray, cam_image: np.ndarray, 
                                save_path: str, colormap: int):
    """
    Saves Grad-CAM visualization as an image.
    Args:
        image (np.ndarray): Original image in NumPy format.
        cam_image (np.ndarray): Grad-CAM mask.
        save_path (str): File path to save the visualization.
        colormap (int): OpenCV colormap to apply.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_image), colormap)
    superimposed_image = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(save_path, superimposed_image)
    print(f"Grad-CAM saved: {save_path}")

def generate_grad_cam(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, 
                      incorrect_indices: list, save_dir: str = DEFAULT_SAVE_DIR):
    """
    Generates Grad-CAM visualizations for incorrectly classified images.
    Args:
        model (nn.Module): Trained model.
        images (torch.Tensor): Tensor of input images.
        labels (torch.Tensor): Tensor of true labels.
        incorrect_indices (list): List of indices for misclassified images.
        save_dir (str): Directory to save the Grad-CAM visualizations.
    """
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # Handle DataParallel wrapper

    # Initialize Grad-CAM extractor
    cam_extractor = GradCAM(model, target_layer=get_target_layer(model, TARGET_LAYER_NAME))

    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

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
        original_image = np.uint8(original_image * 255)

        # Save Grad-CAM visualizations with different colormaps
        for cmap_name, cmap in COLORMAPS.items():
            output_filename = os.path.join(
                save_dir, f'grad_cam_true_{label}_pred_{pred_class}_{idx}_{cmap_name}.png'
            )
            save_grad_cam_visualization(original_image, cam_image, output_filename, cmap)

def main():
    """
    Main function to execute Grad-CAM generation.
    """
    from models import LResNet50E_IR  # Placeholder for your model import

    # Initialize the model
    num_classes = 7
    model = LResNet50E_IR(num_classes=num_classes).to(DEVICE)
    model.eval()

    # List model layers for inspection
    list_model_layers(model)

    # Get target layer and print confirmation
    target_layer = get_target_layer(model, TARGET_LAYER_NAME)
    print(f"Target layer: {target_layer}")

    # Example input tensor
    input_tensor = torch.randn(1, 3, 224, 224).to(DEVICE)

    # Test layer hook
    hook, get_activations = register_layer_hook(target_layer)
    with torch.no_grad():
        _ = model(input_tensor)
    activations = get_activations()

    if activations is not None:
        print(f"Activations captured: Shape {activations.shape}")
    else:
        print("No activations captured. Verify the target layer.")

    hook.remove()

if __name__ == "__main__":
    main()
