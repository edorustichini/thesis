import pandas as pd
from sklearn.svm import SVC
from manager import GridSearch, SVM, RandomizedSearch
from tester import test_process
from trainer import train_grid_search, train_random_search, train_model_no_search

import sys
sys.path.append('../')
from common import load_on_RAM
from config import setup_parser
from data.preprocessing import (
    flatten_latents_YUV,
    Y_all_patches_per_latent, 
    Y_multiple_patches_per_latent, 
    Y_single_patch_per_latent, 
    YUV_all_patches_per_latent, 
    YUV_single_patch_per_latent,
    YUV_multiple_patches_per_latent
)

def train_SVM_grid(args, preprocess):
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto'],
        'degree': [2, 3, 4],
        'probability': [True],
        'shrinking': [True, False]
    }
    svm = SVM(random_state=42, probability=True)
    train_grid_search(args, svm, param_grid, preprocess)

def train_SVM_random(args, preprocess):
    import scipy.stats as stats
    
    param_distributions = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto'],
        'degree': [2, 3],
        'shrinking': [True, False]
    }
    args.model_name = "SVM_random_search"
    svm = SVC(random_state=42, probability=True)
    train_random_search(args, svm, param_distributions, preprocess, n_iter=30)

def print_search_results(search_results):
    results = pd.DataFrame(search_results.cv_results_)
    params_df = results["params"].apply(pd.Series)
    summary = pd.concat([results[["mean_test_score", "std_test_score"]], params_df], axis=1)
    summary = summary.sort_values(by="mean_test_score", ascending=False)
    print(summary[["mean_test_score", "std_test_score", "kernel", "C"]].head(5))

############### EXPERIMENTS ###############

def exp_single_Y_SVM_random_search(args):
    args.models_save_dir = "/data/lesc/users/rustichini/thesis/models_saved/SINGLE_PATCH/Y/NEW"
    
    train_SVM_random(args, preprocess=Y_single_patch_per_latent)
    
    model_y = load_on_RAM(args.model_file_y)
    model_y_hat = load_on_RAM(args.model_file_y_hat)
    print("Results for model trained on y")
    print_search_results(model_y)
    print("Results for model trained on y_hat")
    print_search_results(model_y_hat)

    test_process(args, preprocess=Y_single_patch_per_latent)

def exp_single_YUV_SVM_random_search(args):
    args.models_save_dir = "/data/lesc/users/rustichini/thesis/models_saved/SINGLE_PATCH/YUV/NEW"
    train_SVM_random(args, preprocess=YUV_single_patch_per_latent)
    model_y = load_on_RAM(args.model_file_y)
    model_y_hat = load_on_RAM(args.model_file_y_hat)
    print("Results for model trained on y")
    print_search_results(model_y)
    print("Results for model trained on y_hat")
    print_search_results(model_y_hat)
    test_process(args, preprocess=YUV_single_patch_per_latent)

def exp_multiple_Y_SVM_random_search(args):
    args.models_save_dir = "/data/lesc/users/rustichini/thesis/models_saved/MULTIPLE_PATCHES/Y/NEW"
    
    train_SVM_random(args, preprocess=Y_multiple_patches_per_latent)
    
    model_y = load_on_RAM(args.model_file_y)
    model_y_hat = load_on_RAM(args.model_file_y_hat)
    print("Results for model trained on y")
    print_search_results(model_y)
    print("Results for model trained on y_hat")
    print_search_results(model_y_hat)
    test_process(args, preprocess=Y_all_patches_per_latent)

def exp_multiple_YUV_SVM_random_search(args):
    args.models_save_dir = "/data/lesc/users/rustichini/thesis/models_saved/MULTIPLE_PATCHES/YUV/NEW"
    train_SVM_random(args, preprocess=YUV_multiple_patches_per_latent)
    model_y = load_on_RAM(args.model_file_y)
    model_y_hat = load_on_RAM(args.model_file_y_hat)
    print("Results for model trained on y")
    print_search_results(model_y)
    print("Results for model trained on y_hat")
    print_search_results(model_y_hat)
    test_process(args, preprocess=YUV_all_patches_per_latent)

def exp_single_Y_SVM_no_search(args):
    args.models_save_dir = "/data/lesc/users/rustichini/thesis/models_saved/SINGLE_PATCH/Y"
    svm = SVM(random_state=42, probability=True, kernel='rbf', C=100, gamma='auto')
    train_model_no_search(args, svm, preprocess=Y_single_patch_per_latent)
    
    test_process(args, preprocess=Y_single_patch_per_latent)

def exp_single_YUV_SVM_no_search(args):
    args.models_save_dir = "/data/lesc/users/rustichini/thesis/models_saved/SINGLE_PATCH/YUV"
    svm = SVM(random_state=42, probability=True, kernel='rbf', C=100, gamma='auto')
    train_model_no_search(args, svm, preprocess=YUV_single_patch_per_latent)
    
    test_process(args, preprocess=YUV_single_patch_per_latent)


if __name__ == "__main__":
    args = setup_parser()
    
    args.save_latents = True
    args.num_samples = 10000
    args.num_samples_test = 1000

    args.set_target_bpp = 1200
    exp_single_YUV_SVM_random_search(args)
    exp_single_Y_SVM_random_search(args)

    args.set_target_bpp = 600
    exp_single_YUV_SVM_random_search(args)
    exp_single_Y_SVM_random_search(args)

    """
    args.num_samples = 30000

    exp_single_Y_SVM_random_search(args)

    args.num_samples = 30000
    args.num_samples_test = 3000
    exp_single_Y_SVM_no_search(args)
    
    #exp_multiple_Y_SVM_random_search(args)
    #exp_multiple_YUV_SVM_random_search(args)
    """
