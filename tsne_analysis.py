import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# Suponha que `embeddings` é uma matriz NxD com N amostras e D dimensões (extraída pelo modelo)
# E `labels` é um vetor com as classes correspondentes

# Reduzindo a dimensionalidade com t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
embeddings_2d = tsne.fit_transform(embeddings)

# Plotando
plt.figure(figsize=(10, 8))
for class_id in np.unique(labels):
    indices = np.where(labels == class_id)
    plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=f'Classe {class_id}', alpha=0.7)

plt.legend()
plt.title("Visualização t-SNE dos Embeddings")
plt.xlabel("Dimensão 1")
plt.ylabel("Dimensão 2")
plt.grid(True)
plt.show()
