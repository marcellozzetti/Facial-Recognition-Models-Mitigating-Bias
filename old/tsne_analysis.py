import torch
from sklearn.manifold import TSNE
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime

def generate_tsne_visualization(model, test_loader, label_encoder, arc_margin):
    model.eval()  # Certifique-se de que o modelo está em modo de avaliação
    embeddings = []
    labels = []
    
    # Certifique-se de que os tensores exigem gradientes antes de registrar o hook
    for images, targets in test_loader:
        images = images.cuda()
        targets = targets.cuda()

        # Garantir que o modelo exige gradientes
        with torch.set_grad_enabled(True):
            # Extração das características do modelo (com gradientes habilitados)
            outputs = model(images)
            embeddings.append(outputs.cpu().detach().numpy())
            labels.append(targets.cpu().detach().numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot the t-SNE results
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        indices = labels == label
        plt.scatter(
            embeddings_2d[indices, 0], 
            embeddings_2d[indices, 1], 
            label=label_encoder.inverse_transform([label])[0], 
            alpha=0.7
        )

    plt.title("t-SNE Visualization of Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    save_path = f'{output_dir}/tsne_visualization_{timestamp}.png'
    plt.savefig(save_path)
    print(f"t-SNE visualization saved to {save_path}")
    plt.show()
