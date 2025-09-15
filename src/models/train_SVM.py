from manager import GridSearch, RandomizedSearch, SVM
from sklearn.svm import SVC
from trainer import train_grid_search, train_random_search, train_model_no_search

import sys
sys.path.append('../')
from config import setup_parser
from data.preprocessing import flatten_latents, create_patches_dataset

def train_SVM_grid(args, preprocess):
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto'],
        'degree': [2, 3, 4],
        'probability': [True]
    }
    svm = SVC(random_state=42, probability=True)
    train_grid_search(args, svm, param_grid, preprocess)

def train_SVM_random(args, preprocess):
    import scipy.stats as stats
    param_distributions = {
        'C': stats.loguniform(1e-2, 1e2),
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto'],
        'degree': stats.randint(2, 5),
        'probability': [True]
    }
    svm = SVC(random_state=42, probability=True)
    train_random_search(args, svm, param_distributions, preprocess)
    


if __name__ == "__main__":
    args = setup_parser()
    # Esempio di chiamata:
    # train_SVM_grid(args, preprocess=create_patches_dataset)
    # train_SVM_random(args, preprocess=create_patches_dataset)

    svm = SVM(random_state=42, probability=True)
    preprocess = create_patches_dataset
    train_SVM_random(args,preprocess=preprocess)

