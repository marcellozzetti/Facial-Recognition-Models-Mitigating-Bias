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

import numpy as np
import cv2
import matplotlib.pyplot as plt

def save_grad_cam_visualization(original_image, cam_image, output_filename, cmap, original_filename):
    """
    Combina a imagem original com a máscara Grad-CAM e salva as visualizações.
    Inclui logs de debug para verificar dimensões e tipos de dados.
    """
    print(f"[DEBUG] Tipo de original_image: {type(original_image)}, Shape: {original_image.shape}")
    print(f"[DEBUG] Tipo de cam_image: {type(cam_image)}, Shape: {cam_image.shape}")
    print(f"[DEBUG] Tipo de cmap: {type(cmap)}")

    # Se a máscara tem 1 canal, converta para 3 canais
    if cam_image.ndim == 2:
        cam_image = np.stack([cam_image] * 3, axis=-1)  # Duplica o canal para RGB
        print(f"[DEBUG] cam_image convertido para 3 canais: Shape {cam_image.shape}")

    # Normalizar a máscara para [0, 1]
    cam_image = (cam_image - cam_image.min()) / (cam_image.max() - cam_image.min() + 1e-8)
    print(f"[DEBUG] cam_image normalizado para range [0, 1]")

    # Aplicar o colormap
    try:
        heatmap = cmap(cam_image[:, :, 0])[:, :, :3]  # Remove o canal alpha do colormap
        print(f"[DEBUG] heatmap gerado com sucesso")
    except Exception as e:
        print(f"[DEBUG] Erro ao aplicar cmap: {e}")
        raise

    # Redimensionar o heatmap para corresponder à imagem original
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    print(f"[DEBUG] heatmap redimensionado para shape {heatmap.shape}")

    # Verifica compatibilidade de canais
    if original_image.shape[-1] != 3 or heatmap.shape[-1] != 3:
        raise ValueError(
            f"Erro na sobreposição: a imagem original tem {original_image.shape[-1]} canais "
            f"e a máscara tem {heatmap.shape[-1]} canais."
        )

    # Combina Grad-CAM com a imagem original (0.5 peso para cada)
    overlay = (0.5 * original_image + 0.5 * (heatmap * 255)).astype(np.uint8)
    print(f"[DEBUG] overlay gerado com sucesso")

    # Salvar imagens
    cv2.imwrite(original_filename, cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(output_filename, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"[DEBUG] Imagens salvas: {original_filename} e {output_filename}")


def generate_grad_cam(model, images, labels, incorrect_indices, label_encoder):
    """
    Gera Grad-CAMs para imagens fornecidas e salva as visualizações.
    Inclui logs de debug para checar tipos de dados e processamento.
    """
    print(f"[DEBUG] Tipo de images: {type(images)}, Shape: {images.shape}")
    print(f"[DEBUG] Tipo de labels: {type(labels)}")
    print(f"[DEBUG] Tipo de incorrect_indices: {type(incorrect_indices)}, Valores: {incorrect_indices}")

    cmap = plt.cm.get_cmap('jet')  # Certifique-se de que cmap é um colormap válido
    print(f"[DEBUG] cmap inicializado como: {cmap}")

    for idx in incorrect_indices:
        original_image = images[idx].cpu().numpy().transpose(1, 2, 0)  # Convert tensor para formato HWC
        cam_image = np.random.rand(224, 224, 3)  # Mock de uma imagem Grad-CAM; substitua pelo real
        label = labels[idx].item()
        predicted_label = 0  # Mock do rótulo previsto; substitua pelo real
        label_name = label_encoder.inverse_transform([label])[0]
        predicted_name = label_encoder.inverse_transform([predicted_label])[0]

        # Gerar nome de arquivo para salvar as visualizações
        original_filename = f"original_{idx}_{label_name}.jpg"
        output_filename = f"gradcam_{idx}_{label_name}_as_{predicted_name}.jpg"

        try:
            save_grad_cam_visualization(
                original_image, cam_image, output_filename, cmap, original_filename
            )
        except Exception as e:
            print(f"[DEBUG] Erro ao salvar Grad-CAM para índice {idx}: {e}")
            continue


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
