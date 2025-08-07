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
    parser = argparse.ArgumentParser(description='Compress a directory of images using the RecoEncoder')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--imgs_path', type=str, default='../../real_vs_fake/real-vs-fake', help='Input directory')
    parser.add_argument('input_path', type=str, default='../../input_imgs', help='Input directory')
    parser.add_argument('bin_path', type=str, default='../../JPEGAI_output/', help='Save directory')
    parser.add_argument('--set_target_bpp', type=int, default=1, help='Set the target bpp '
                                                                      '(multiplied by 100)')
    parser.add_argument('--models_dir_name', type=str, default='../../jpeg-ai-reference-software/models', help='Directory name for the '
                                                                                 'models used in the encoder-decoder'
                                                                                 'pipeline')
    #Arguments for training
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to train on')
    parser.add_argument('--num_samples_test', type=int, default=300, help='Number of samples to test on')
    parser.add_argument('--random_sample', type=bool, default=False, help='Sample')
    parser.add_argument("--train_csv",default="../../train.csv" , help="Path to dataset's csv file")
    parser.add_argument("--test_csv", default="../../test.csv", help="Path to test's csv file")
    parser.add_argument("-t", "--target",default=None, help="y_hat if quantized latent, else y")
    parser.add_argument("--save", default=False, help="True if wanted to save dataset")
    parser.add_argument("--models_save_dir", default="/data/lesc/users/rustichini/thesis/models_saved", help="Directory to save models")

    args = parser.parse_args()
    
    X, X_hat, y = create_dataset()
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


