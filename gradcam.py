from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image

def generate_gradcam_visualization(model, target_layer, loader, device, output_dir="output/gradcam"):
    os.makedirs(output_dir, exist_ok=True)

    # Grad-CAM instance
    cam_extractor = SmoothGradCAMpp(model, target_layer)
    
    model.eval()  # Certifique-se de que o modelo está em modo de avaliação
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass para gerar predições
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            for idx in range(min(5, len(images))):  # Limite a visualização para 5 amostras
                img = images[idx]
                label = labels[idx].item()
                pred = preds[idx].item()
                
                # Gerar Grad-CAM para a predição
                activation_map = cam_extractor(pred, outputs[idx].unsqueeze(0))
                
                # Overlay da ativação na imagem original
                heatmap = to_pil_image(activation_map[0].squeeze(0))
                input_image = to_pil_image(img.cpu())
                result = overlay_mask(input_image, heatmap, alpha=0.5)
                
                # Salvar a visualização
                result.save(os.path.join(output_dir, f"batch{batch_idx}_sample{idx}_true{label}_pred{pred}.png"))
            
            # Limite apenas ao primeiro batch para evitar muitas imagens
            if batch_idx == 0:
                break
