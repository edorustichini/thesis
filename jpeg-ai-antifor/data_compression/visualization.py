from train_RF import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from train_RF import load, save

def visualize(X, y, alg, reduce: bool=False):
    if reduce:
        #TODO add PCA
        pass
    X_embedded = TSNE(n_components=2, perplexity=60, random_state=42, n_jobs=4, learning_rate=50)
    

def main():
    dataset_path = "/data/lesc/users/rustichini/thesis/models_saved/6_bpp/15000_samples/y/dataset.joblib"
    X,y = load(dataset_path)
    print("Dataset loaded")
    #print("PCA begins")
    #print("PCA finished")
    print("TSNE begins")
    X_embedded = TSNE(n_components=2, random_state=42, n_jobs=4).fit_transform(X)
    print("TSNE finished")
    print(f"Shape of X_embedded: {X_embedded.shape}")
    data = {}
    data['x'] = X_embe
    sns.scatterplot

   ''' plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=10, alpha=0.7)

    legend = plt.legend(*scatter.legend_elements(), title="Class")
    plt.gca().add_artist(legend)

    plt.title("Proiezione t-SNE del Dataset (colorata per classe)")
    plt.xlabel("Componente t-SNE 1")
    plt.ylabel("Componente t-SNE 2")
    plt.grid(True)
    plt.tight_layout() # Adatta automaticamente i parametri del plot per un layout stretto

    # Salva il grafico prima di mostrarlo (plt.show() potrebbe chiuderlo)
    plt.savefig("tsne_projection.png", dpi=300)
    print("Plot saved as tsne_projection.png")

    # Mostra il grafico
    plt.show()
'''


if __name__ == "__main__":
    main()


