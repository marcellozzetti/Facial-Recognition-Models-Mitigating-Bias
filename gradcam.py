import os
import torch
import torchcam
from torchcam.methods import GradCAM
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch.nn as nn

# Nome da camada alvo
target_layer_name = 'backbone.layer4.2.conv3'

# Verifica se CUDA está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Captura a referência ao módulo da camada alvo
def obter_target_layer(model, layer_name):
    target_layer = dict(model.named_modules()).get(layer_name, None)
    if target_layer is None:
        raise ValueError(f"Camada alvo '{layer_name}' não encontrada no modelo. Verifique os nomes listados acima.")
    return target_layer

def generate_grad_cam(model, images, labels, incorrect_indices, save_dir='output/grad_cam'):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    cam_extractor = GradCAM(model, target_layer=obter_target_layer(model, target_layer_name))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx in incorrect_indices:
        image = images[idx].unsqueeze(0).to(device)  # Imagem para o modelo
        label = labels[idx].unsqueeze(0).to(device)  # Rótulo da imagem

        # Calcular a previsão
        output = model(image)  # Obtenção da saída do modelo (logits ou probabilidades)
        pred_class = output.squeeze(0).argmax().item()  # Classe predita
        true_class = label.item()  # Classe correta

        # Certificar-se de que pred_class é um índice válido
        if pred_class < 0 or pred_class >= output.size(1):  # Verifica se pred_class está no intervalo de classes
            raise ValueError(f"Predicted class index {pred_class} is out of bounds for the output size {output.size(1)}")

        # Não é necessário passar `input_tensor` como argumento diretamente
        cam = cam_extractor(class_idx=pred_class, scores=output)  # Passando apenas as pontuações e índice da classe

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
