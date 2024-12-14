import torch
from sklearn.manifold import TSNE
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Function to generate t-SNE visualization
def generate_tsne_visualization(model, data_loader, label_encoder, arc_face_margin=None):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for images, batch_labels in tqdm(data_loader):
            images = images.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(images)
            if arc_face_margin is not None:
                outputs = arc_face_margin(outputs, batch_labels)
            
            embeddings.append(outputs.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())

    # Convert lists to arrays
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plotting
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        indices = labels == label
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=label_encoder.inverse_transform([label])[0], alpha=0.7)

    plt.title("t-SNE Visualization of Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'output/tsne_visualization_{timestamp}.png')
    plt.show()
