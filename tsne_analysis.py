import torch
from sklearn.manifold import TSNE
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime

def generate_tsne_visualization(model, data_loader, label_encoder, arc_face_margin=None, output_dir='output'):
    """
    Generate a 2D t-SNE visualization of model embeddings.

    Args:
        model (torch.nn.Module): The trained model for feature extraction.
        data_loader (torch.utils.data.DataLoader): DataLoader providing images and labels.
        label_encoder (sklearn.preprocessing.LabelEncoder): Encoder to decode class labels.
        arc_face_margin (callable, optional): Margin layer for ArcFace, if applicable.
        output_dir (str): Directory to save the t-SNE visualization.
    """
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')

    # Set model to evaluation mode
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():  # Ensure no gradients are computed
        for images, batch_labels in tqdm(data_loader, desc="Extracting embeddings"):
            images = images.to(device)
            batch_labels = batch_labels.to(device)

            # Remove hooks if they're being registered by torchcam
            model.zero_grad()  # Ensure no unnecessary hooks are added

            outputs = model(images)

            # Detach if necessary to avoid errors
            if isinstance(outputs, torch.Tensor) and not outputs.requires_grad:
                outputs = outputs.detach()

            if arc_face_margin is not None:
                outputs = arc_face_margin(outputs, batch_labels)

            embeddings.append(outputs.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())

    # Stack embeddings and labels into arrays
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

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
