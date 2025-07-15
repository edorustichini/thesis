from train_RF import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from train_RF import load, save

def main():
    '''dataset_path = "/data/lesc/users/rustichini/thesis/models_saved/6_bpp/15000_samples/y/dataset.joblib"
    X,y = load(dataset_path)
    print("Dataset loaded")
    print("PCA begins")
    pca = PCA(n_components=50)
    print("PCA finished")
    print("TSNE begins")
    X_reduced = pca.fit_transform(X)
    X_embedded = TSNE().fit_transform(X_reduced)
    print(X_embedded.shape)
    save((X_embedded, y),"/data/lesc/users/rustichini/thesis/models_saved/6_bpp/15000_samples/y/", "embedded_dataset") 
    print("Finished")'''
    X,y = load("/data/lesc/users/rustichini/thesis/models_saved/6_bpp/15000_samples/y/embedded_dataset.joblib")

            # X: array (15000, 2) -> output di TSNE
# y: array (15000,) -> etichette (es. 0 o 1 per real/fake)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=10, alpha=0.7)

# Aggiungi legenda per le classi
    legend = plt.legend(*scatter.legend_elements(), title="Class")
    plt.gca().add_artist(legend)

    plt.title("t-SNE projection colored by class")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("tsne_projection.png", dpi=300)


if __name__ == "__main__":
    main()


