import os
import torch
import torch.nn as nn
from torchcam.methods import GradCAM
import cv2
import numpy as np

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Target layer name
TARGET_LAYER_NAME = 'backbone.layer4.2.conv3'

def get_target_layer(model, layer_name):
    """
    Retrieves the target layer from the model.
    Args:
        model: The neural network model.
        layer_name: The name of the target layer.
    Returns:
        The target layer module.
    """
    target_layer = dict(model.named_modules()).get(layer_name)
    if target_layer is None:
        raise ValueError(f"Target layer '{layer_name}' not found in the model.")
    return target_layer

def list_model_layers(model):
    """
    Lists all the layers in the model with their names.
    Args:
        model: The neural network model.
    """
    print("Listing model layers:")
    for name, module in model.named_modules():
        print(name)

def register_layer_hook(layer):
    """
    Registers a forward hook to capture activations of a given layer.
    Args:
        layer: Target layer for activation capture.
    Returns:
        forward_hook: The registered forward hook.
    """
    activations = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    hook = layer.register_forward_hook(forward_hook)
    return hook, lambda: activations

def generate_grad_cam(model, images, labels, incorrect_indices, save_dir='output/grad_cam'):
    """
    Generates Grad-CAM visualizations for incorrectly classified images.
    
    Args:
        model: Trained model.
        images: Tensor of input images.
        labels: Tensor of true labels.
        incorrect_indices: List of indices for misclassified images.
        save_dir: Directory to save the Grad-CAM visualizations.
    """
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # Handle DataParallel wrapper

    # Initialize Grad-CAM extractor
    cam_extractor = GradCAM(model, target_layer=get_target_layer(model, TARGET_LAYER_NAME))

    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Define colormaps
    colormaps = {
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "cool": cv2.COLORMAP_COOL,
        "rainbow": cv2.COLORMAP_RAINBOW
    }

    for idx in incorrect_indices:
        image = images[idx].unsqueeze(0).to(device)
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

        # Generate and save Grad-CAM visualizations with different colormaps
        for cmap_name, cmap in colormaps.items():
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_image), cmap)
            superimposed_image = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

            # Save image with filename indicating class info
            output_filename = os.path.join(
                save_dir, f'grad_cam_true_{label}_pred_{pred_class}_{idx}_{cmap_name}.png'
            )
            cv2.imwrite(output_filename, superimposed_image)
            print(f"Grad-CAM saved: {output_filename}")
