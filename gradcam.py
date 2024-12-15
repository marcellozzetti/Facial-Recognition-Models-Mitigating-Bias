import os
import torchcam
from torchcam.methods import GradCAM
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn as nn

# Função para gerar e salvar a visualização de Grad-CAM
def generate_grad_cam(model, images, labels, incorrect_indices, save_dir='output/grad_cam'):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    
    cam_extractor = GradCAM(model, target_layers='')  # Altere a camada conforme necessário
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx in incorrect_indices:
        image = images[idx].unsqueeze(0).to(device)
        label = labels[idx].unsqueeze(0).to(device)

        # Calcular a previsão e aplicar o Grad-CAM
        output = model(image)
        pred_class = output.squeeze(0).argmax().item()  # Classe predita
        true_class = label.item()  # Classe correta

        # Calcular Grad-CAM para a classe predita
        cam = cam_extractor(pred_class, output)

        # Converta a imagem para numpy e aplique a máscara Grad-CAM
        cam_image = cam.squeeze().cpu().numpy()
        cam_image = cv2.resize(cam_image, (image.shape[3], image.shape[2]))  # Redimensiona a máscara para o tamanho da imagem
        cam_image = np.maximum(cam_image, 0)  # Garante que não haja valores negativos
        cam_image = cam_image / cam_image.max()  # Normaliza para 0-1

        # Convert the original image to numpy (for display)
        original_image = image.squeeze().cpu().numpy().transpose((1, 2, 0))
        original_image = np.uint8(original_image * 255)  # Transformação para imagem

        # Aplica a máscara Grad-CAM à imagem original
        heatmap = np.uint8(255 * cam_image)  # Gera o heatmap
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Aplica o mapa de cores
        superimposed_image = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

        # Nome do arquivo com a classe correta e predita
        output_filename = os.path.join(save_dir, f'grad_cam_true_{true_class}_pred_{pred_class}_{idx}.png')
        cv2.imwrite(output_filename, superimposed_image)
        print(f"Grad-CAM image saved as {output_filename}")
