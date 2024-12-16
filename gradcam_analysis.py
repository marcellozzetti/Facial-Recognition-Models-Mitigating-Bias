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

def save_grad_cam_visualization(original_image, cam_image, output_filename, cmap, original_filename):
    """
    Combina a imagem original com a máscara Grad-CAM e salva as visualizações.
    """
    # Se a máscara tem 1 canal, converta para 3 canais
    if cam_image.ndim == 2:
        cam_image = np.stack([cam_image] * 3, axis=-1)  # Duplica o canal para RGB

    # Normalizar a máscara para [0, 1]
    cam_image = (cam_image - cam_image.min()) / (cam_image.max() - cam_image.min() + 1e-8)

    # Aplicar o colormap
    heatmap = cmap(cam_image[:, :, 0])[:, :, :3]  # Remove o canal alpha do colormap

    # Redimensionar o heatmap para corresponder à imagem original
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))

    # Verifica compatibilidade de canais
    if original_image.shape[-1] != 3 or heatmap.shape[-1] != 3:
        raise ValueError(
            f"Erro na sobreposição: a imagem original tem {original_image.shape[-1]} canais e a máscara tem {heatmap.shape[-1]} canais."
        )

    # Combina Grad-CAM com a imagem original (0.5 peso para cada)
    overlay = (0.5 * original_image + 0.5 * (heatmap * 255)).astype(np.uint8)

    # Salvar imagens
    cv2.imwrite(original_filename, cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(output_filename, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

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
        if isinstance(cam_image, list):  # Handle list return
            cam_image = cam_image[0]
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
