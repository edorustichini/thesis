import os
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from manager import LGBM, GridSearch, RandomizedSearch
from tester import test_process, test_single_target
from trainer import train_grid_search, train_random_search, train_model_no_search

import sys
sys.path.append('../')
from data.dataset_manager import prepare_dataset

from common import load_on_RAM
from config import setup_parser
from data.preprocessing import (
    flatten_latents_Y,
    flatten_latents_YUV,
    Y_all_patches_per_latent, 
    Y_multiple_patches_per_latent, 
    Y_single_patch_per_latent, 
    YUV_all_patches_per_latent, 
    YUV_single_patch_per_latent,
    YUV_multiple_patches_per_latent
)

def train_LGBM_grid(args, preprocess, X_raw=None, X_hat_raw=None, labels=None):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 20, -1],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 50, 100],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    lgbm = HistGradientBoostingClassifier(random_state=42, verbose=-1)
    train_grid_search(args, lgbm, param_grid, preprocess, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)

def train_LGBM_random(args, preprocess, X_raw=None, X_hat_raw=None, labels=None):
    import scipy.stats as stats
    param_distributions = {
    'learning_rate': stats.uniform(0.01, 0.29),       # shrinkage rate
    'n_estimators': stats.randint(100, 600),          # numero di boosting rounds
    'num_leaves': stats.randint(15, 129),             # foglie per albero
    'reg_lambda': stats.loguniform(1e-6, 1.0),        # regularizzazione L2
    'min_child_samples': stats.randint(1, 51),        # min dati per foglia
    'subsample': stats.uniform(0.5, 0.5),             # row sampling
    'colsample_bytree': stats.uniform(0.5, 0.5) 
    }
    args.model_name = "GB_random_search"
    lgbm = HistGradientBoostingClassifier(random_state=42)
    train_random_search(args, lgbm, param_distributions, preprocess, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)

def train_LGBM_no_search(args, preprocess,lgbm=None, X_raw=None, X_hat_raw=None, labels=None):
    lgbm = LGBM(random_state=42, verbose=0) if lgbm is None else lgbm
    train_model_no_search(args, lgbm, preprocess, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)

def print_search_results(search_results):
    results = pd.DataFrame(search_results.cv_results_)
    
    # "params" contiene i dizionari dei parametri provati
    params_df = results["params"].apply(pd.Series)
    
    # concateniamo score + parametri
    summary = pd.concat(
        [results[["mean_test_score", "std_test_score"]], params_df],
        axis=1
    )
    
    # ordiniamo per score
    summary = summary.sort_values(by="mean_test_score", ascending=False)
    
    # colonne importanti per HistGradientBoosting
    cols = ["mean_test_score", "std_test_score"]
    for c in ["max_depth", "max_iter", "learning_rate", "max_leaf_nodes", 
              "l2_regularization", "min_samples_leaf", "max_bins"]:
        if c in summary.columns:
            cols.append(c)

    print(summary[cols].head(5))

############### EXPERIMENTS ###############

def exp_single_Y_LGBM_random_search(args, X_raw=None, X_hat_raw=None, labels=None):
    args.models_save_dir = "/data/lesc/users/rustichini/thesis/models_saved/SINGLE_PATCH/Y/LGBM"
    train_LGBM_random(args, preprocess=Y_single_patch_per_latent, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)
    model_y = load_on_RAM(args.model_file_y)
    model_y_hat = load_on_RAM(args.model_file_y_hat)
    print("Results for model trained on y")
    print_search_results(model_y)
    print("Results for model trained on y_hat")
    print_search_results(model_y_hat)
    test_process(args, preprocess=Y_single_patch_per_latent)

def exp_single_YUV_LGBM_random_search(args, X_raw=None, X_hat_raw=None, labels=None):
    args.models_save_dir = "/data/lesc/users/rustichini/thesis/models_saved/SINGLE_PATCH/YUV/LGBM"
    train_LGBM_random(args, preprocess=YUV_single_patch_per_latent, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)
    model_y = load_on_RAM(args.model_file_y)
    model_y_hat = load_on_RAM(args.model_file_y_hat)
    print("Results for model trained on y")
    print_search_results(model_y)
    print("Results for model trained on y_hat")
    print_search_results(model_y_hat)
    test_process(args, preprocess=YUV_single_patch_per_latent)

def exp_multiple_Y_LGBM_random_search(args, X_raw=None, X_hat_raw=None, labels=None):
    args.models_save_dir = "/data/lesc/users/rustichini/thesis/models_saved/MULTIPLE_PATCHES/Y/"
    train_LGBM_random(args, preprocess=Y_multiple_patches_per_latent, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)
    model_y = load_on_RAM(args.model_file_y)
    model_y_hat = load_on_RAM(args.model_file_y_hat)
    print("Results for model trained on y")
    print_search_results(model_y)
    print("Results for model trained on y_hat")
    print_search_results(model_y_hat)
    test_process(args, preprocess=Y_all_patches_per_latent)

def exp_multiple_YUV_LGBM_random_search(args, X_raw=None, X_hat_raw=None, labels=None):
    args.models_save_dir = "/data/lesc/users/rustichini/thesis/models_saved/MULTIPLE_PATCHES/YUV/"
    train_LGBM_random(args, preprocess=YUV_multiple_patches_per_latent, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)
    model_y = load_on_RAM(args.model_file_y)
    model_y_hat = load_on_RAM(args.model_file_y_hat)
    print("Results for model trained on y")
    print_search_results(model_y)
    print("Results for model trained on y_hat")
    print_search_results(model_y_hat)
    test_process(args, preprocess=YUV_all_patches_per_latent)


def exp_single_Y_LGBM_no_search(args,lgbm, X_raw=None, X_hat_raw=None, labels=None):
    args.models_save_dir = "/data/lesc/users/rustichini/thesis/models_saved/SINGLE_PATCH/Y/NO_SEARCH_LGBM"
    train_LGBM_no_search(args, preprocess=Y_single_patch_per_latent, lgbm=lgbm, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)
    test_process(args, preprocess=Y_single_patch_per_latent)

def exp_single_YUV_LGBM_no_search(args,lgbm, X_raw=None, X_hat_raw=None, labels=None):
    args.models_save_dir = "/data/lesc/users/rustichini/thesis/models_saved/SINGLE_PATCH/YUV/NO_SEARCH_LGBM"
    train_LGBM_no_search(args, preprocess=YUV_single_patch_per_latent, lgbm=lgbm, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)
    args.model_file_y = "/data/lesc/users/rustichini/thesis/models_saved/SINGLE_PATCH/YUV/600_bpp/30000_samples/y/RF_463estimators.joblib"
    args.model_file_y_hat = "/data/lesc/users/rustichini/thesis/models_saved/SINGLE_PATCH/YUV/600_bpp/30000_samples/y_hat/RF_463estimators.joblib"
    test_process(args, preprocess=YUV_single_patch_per_latent)

def exp_flatten_YUV_GB_no_search(args, lgbm=None, X_raw=None, X_hat_raw=None, labels=None):
    args.models_save_dir = "/data/lesc/users/rustichini/thesis/models_saved/FLATTEN/"
    train_LGBM_no_search(args, preprocess=flatten_latents_YUV, lgbm=lgbm, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)
    test_process(args, preprocess=flatten_latents_YUV)

def exp_flatten_Y_GB_no_search(args, lgbm=None, X_raw=None, X_hat_raw=None, labels=None):
    args.models_save_dir = "/data/lesc/users/rustichini/thesis/models_saved/FLATTEN/Y/"
    train_LGBM_no_search(args, preprocess=flatten_latents_Y, lgbm=lgbm, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)
    test_process(args, preprocess=flatten_latents_Y)

def exp_multiple_Y_LGBM_no_search(args, lgbm=None, X_raw=None, X_hat_raw=None, labels=None):
    args.models_save_dir = "/data/lesc/users/rustichini/thesis/models_saved/MULTIPLE_PATCHES/Y/NO_SEARCH_LGBM"
    train_LGBM_no_search(args, preprocess=Y_multiple_patches_per_latent, lgbm=lgbm, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)
    test_process(args, preprocess=Y_all_patches_per_latent)

def exp_multiple_YUV_LGBM_no_search(args, lgbm=None, X_raw=None, X_hat_raw=None, labels=None):
    args.models_save_dir = "/data/lesc/users/rustichini/thesis/models_saved/MULTIPLE_PATCHES/YUV/NO_SEARCH_LGBM"
    train_LGBM_no_search(args, preprocess=YUV_multiple_patches_per_latent, lgbm=lgbm, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)
    test_process(args, preprocess=YUV_all_patches_per_latent)

def all_exp_LGBM_single(args):
    gb = LGBM(learning_rate=0.12965912630431367,
              max_iter=596,
              max_leaf_nodes=117,
              min_samples_leaf=18,
              l2_regularization=0.003042734607209594,
              random_state=42,
              early_stopping=True,       
              validation_fraction=0.1)
    for bpp in [600,1200]:
        args.set_target_bpp = bpp
        save_latent_path = os.path.join(args.bin_path,str(args.set_target_bpp)+"_bpp","latent") if args.save_latents else None
        X_raw, X_hat_raw, labels = None, None, None #prepare_dataset(args, args.train_csv, save_latent_path)

        args.num_samples = 20000
        #exp_single_Y_LGBM_no_search(args, lgbm=gb, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)
        exp_single_YUV_LGBM_no_search(args, lgbm=gb, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)
        exp_flatten_YUV_GB_no_search(args, lgbm=gb, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)

def all_exp_LGBM_multiple(args):
    gb = LGBM(learning_rate=0.12965912630431367,
              max_iter=596,
              max_leaf_nodes=117,
              min_samples_leaf=18,
              l2_regularization=0.003042734607209594,
              random_state=42,
              early_stopping=True,       # consigliato per risparmiare tempo
              validation_fraction=0.1)
    
    for bpp in [600,1200]:
        args.set_target_bpp = bpp
        save_latent_path = os.path.join(args.bin_path,str(args.set_target_bpp)+"_bpp","latent") if args.save_latents else None
        X_raw, X_hat_raw, labels = prepare_dataset(args, args.train_csv, save_latent_path)
        
        exp_multiple_Y_LGBM_no_search(args, lgbm=gb, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)
        exp_multiple_YUV_LGBM_no_search(args, lgbm=gb, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)

def all_exp_GB_flatten_no_search(args):
    gb = LGBM(learning_rate=0.12965912630431367,
              max_iter=596,
              max_leaf_nodes=117,
              min_samples_leaf=18,
              l2_regularization=0.003042734607209594,
              random_state=42,
              early_stopping=True,       
              validation_fraction=0.1)    
    for bpp in [600, 1200]:
        args.set_target_bpp = bpp
        save_latent_path = os.path.join(args.bin_path, str(args.set_target_bpp)+"_bpp", "latent") if args.save_latents else None
        X_raw, X_hat_raw, labels = prepare_dataset(args, args.train_csv, save_latent_path)
        exp_flatten_Y_GB_no_search(args, gb, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)
        exp_flatten_YUV_GB_no_search(args, gb, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)

def test_model(args, filepath, preprocess):
    model = load_on_RAM(filepath)
    test_process(args, model=model, preprocess=preprocess)

if __name__ == "__main__":
    args = setup_parser()
    print(f"{10*'*'} EXPERIMENTS ON GB {10*'*'}")
    args.save_latents = True
    args.set_target_bpp = 1200
    args.num_samples = 30000
    args.num_samples_test = 3000


    #all_exp_LGBM_single(args)
    #all_exp_GB_flatten_no_search(args)
    all_exp_LGBM_multiple(args)
    
    """
    exp_flatten_GB_no_search(args)
    args.set_target_bpp = 600
    args.num_samples = 35000
    args.num_samples_test = 4000
    exp_flatten_GB_no_search(args)
    
    
    exp_multiple_Y_LGBM_random_search(args)
    exp_multiple_YUV_LGBM_random_search(args)
    args.set_target_bpp = 600
    exp_multiple_Y_LGBM_random_search(args)
    exp_multiple_YUV_LGBM_random_search(args)

    """

    """
    bpp = 1200
    args.set_target_bpp = bpp
    #args.num_samples = 30000
    args.num_samples_test = 3000
    
    save_latent_path = os.path.join(args.bin_path,str(args.set_target_bpp)+"_bpp","test","latent")
    X_raw_test, X_hat_raw_test, labels = prepare_dataset(args, args.test_csv, save_latent_path)
    preprocess = flatten_latents
    model_y_path = f"/data/lesc/users/rustichini/thesis/models_saved/FLATTEN/{bpp}_bpp/35000_samples/y/GB.joblib"
    model_y_hat_path = f"/data/lesc/users/rustichini/thesis/models_saved/FLATTEN/{bpp}_bpp/35000_samples/y_hat/GB.joblib"
    test_single_target(X_raw_test, labels, model_y_path, preprocess, target='y')
    test_single_target(X_hat_raw_test, labels, model_y_hat_path, preprocess, target='y_hat')
    """
    
