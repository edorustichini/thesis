import os
import pandas as pd
from manager import GridSearch, RandomForest, RandomizedSearch
from tester import test_process
from sklearn.ensemble import RandomForestClassifier
from trainer import learn_curve_process, learning_curve, train_grid_search, train_random_search, train_model_no_search

import sys
sys.path.append('../')
from common import load_on_RAM
from data.dataset_manager import prepare_dataset
from config import setup_parser
from data.preprocessing import (flatten_latents,
                                Y_all_patches_per_latent, 
                                Y_multiple_patches_per_latent, 
                                Y_single_patch_per_latent, 
                                YUV_all_patches_per_latent, 
                                YUV_single_patch_per_latent,
                                YUV_multiple_patches_per_latent)

def train_RF_grid(args, preprocess, X_raw=None, X_hat_raw=None, labels=None):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    rf = RandomForestClassifier(random_state=42, oob_score=True, verbose=0)
    train_grid_search(args, rf, param_grid, preprocess, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)

def train_RF_random(args, preprocess, X_raw=None, X_hat_raw=None, labels=None):
    import scipy.stats as stats
    
    param_distributions = {
        'n_estimators': stats.randint(100, 500),
        'max_depth': stats.randint(10, 35),
        'min_samples_leaf': stats.randint(2, 15),
        'max_features': ['sqrt'],
        'bootstrap': [True]
    }
    
    args.model_name = "RF_random_search"

    rf = RandomForestClassifier(random_state=42, oob_score=True, verbose=0)
    train_random_search(args, rf, param_distributions, preprocess, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)

def print_search_results(search_results):
    results = pd.DataFrame(search_results.cv_results_)
    params_df = results["params"].apply(pd.Series)
    summary = pd.concat([results[["mean_test_score", "std_test_score"]], params_df], axis=1)
    summary = summary.sort_values(by="mean_test_score", ascending=False)
    # Mostra solo max_depth e n_estimators
    print(summary[["mean_test_score", "std_test_score", "max_depth", "n_estimators"]].head(5))

def train_RF_no_search(args, rf=None, preprocess=None, X_raw=None, X_hat_raw=None, labels=None):
    rf = RandomForestClassifier(random_state=42, oob_score=True, verbose=0, n_estimators=300, max_depth=20, max_features='sqrt', min_samples_leaf=2) if rf is None else rf
    train_model_no_search(args, rf, preprocess, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)

############### EXPERIMENTS ###############

def exp_single_Y_RF_random_search(args, X_raw=None, X_hat_raw=None, labels=None):
    args.models_save_dir = "/data/lesc/users/rustichini/thesis/models_saved/SINGLE_PATCH/Y/NEW"

    train_RF_random(args, preprocess=Y_single_patch_per_latent, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)

    model_y = load_on_RAM(args.model_file_y)
    model_y_hat = load_on_RAM(args.model_file_y_hat)
    print("Results for model trained on y")
    print_search_results(model_y)
    print("Results for model trained on y_hat")
    print_search_results(model_y_hat)

    test_process(args, preprocess=Y_single_patch_per_latent)


def exp_single_YUV_RF_random_search(args, X_raw=None, X_hat_raw=None, labels=None):
    args.models_save_dir = "/data/lesc/users/rustichini/thesis/models_saved/SINGLE_PATCH/YUV/NEW"

    train_RF_random(args, preprocess=YUV_single_patch_per_latent, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels) # MAIN FUNCTION

    model_y = load_on_RAM(args.model_file_y)
    model_y_hat = load_on_RAM(args.model_file_y_hat)
    print("Results for model trained on y")
    print_search_results(model_y)
    print("Results for model trained on y_hat")
    print_search_results(model_y_hat)

    test_process(args, preprocess=YUV_single_patch_per_latent)

def exp_multiple_YUV_RF_random_search(args, X_raw=None, X_hat_raw=None, labels=None):
    args.models_save_dir = "/data/lesc/users/rustichini/thesis/models_saved/MULTIPLE_PATCHES/YUV/NEW"

    train_RF_random(args, preprocess=YUV_multiple_patches_per_latent, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels) # MAIN FUNCTION

    model_y = load_on_RAM(args.model_file_y)
    model_y_hat = load_on_RAM(args.model_file_y_hat)
    print("Results for model trained on y")
    print_search_results(model_y)
    print("Results for model trained on y_hat")
    print_search_results(model_y_hat)

    test_process(args, preprocess=YUV_all_patches_per_latent) # FIX

def exp_multiple_Y_RF_random_search(args, X_raw=None, X_hat_raw=None, labels=None):
    args.models_save_dir = "/data/lesc/users/rustichini/thesis/models_saved/MULTIPLE_PATCHES/Y/NEW"

    train_RF_random(args, preprocess=Y_multiple_patches_per_latent, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels) # MAIN FUNCTION

    model_y = load_on_RAM(args.model_file_y)
    model_y_hat = load_on_RAM(args.model_file_y_hat)
    print("Results for model trained on y")
    print_search_results(model_y)
    print("Results for model trained on y_hat")
    print_search_results(model_y_hat)

    test_process(args, preprocess=Y_all_patches_per_latent) # FIX


def exp_multiple_Y_RF_no_search(args, rf=None, X_raw=None, X_hat_raw=None, labels=None):
    args.models_save_dir = "/data/lesc/users/rustichini/thesis/models_saved/MULTIPLE_PATCHES/Y/"
    rf = RandomForest(random_state=42, oob_score=True, verbose=0, n_estimators=440, max_depth=30, max_features='sqrt', min_samples_leaf=2) if rf is None else rf
    train_model_no_search(args, rf, preprocess=Y_multiple_patches_per_latent, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)
    test_process(args, preprocess=Y_multiple_patches_per_latent)

def exp_multiple_YUV_RF_no_search(args, rf=None, X_raw=None, X_hat_raw=None, labels=None):
    args.models_save_dir = "/data/lesc/users/rustichini/thesis/models_saved/MULTIPLE_PATCHES/YUV/"
    rf = RandomForest(random_state=42, oob_score=True, verbose=0, n_estimators=440, max_depth=30, max_features='sqrt', min_samples_leaf=2) if rf is None else rf
    train_model_no_search(args, rf, preprocess=YUV_multiple_patches_per_latent, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)
    test_process(args, preprocess=YUV_multiple_patches_per_latent)


def exp_single_Y_RF_no_search(args, rf=None, X_raw=None, X_hat_raw=None, labels=None):
    args.models_save_dir = "/data/lesc/users/rustichini/thesis/models_saved/SINGLE_PATCH/Y/"
    rf = RandomForest(random_state=42, oob_score=True, verbose=0, n_estimators=440, max_depth=30, max_features='sqrt', min_samples_leaf=2) if rf is None else rf
    train_model_no_search(args, rf, preprocess=Y_single_patch_per_latent, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)
    test_process(args, preprocess=Y_single_patch_per_latent)
    
def exp_single_YUV_RF_no_search(args, rf=None, X_raw=None, X_hat_raw=None, labels=None):
    args.models_save_dir = "/data/lesc/users/rustichini/thesis/models_saved/SINGLE_PATCH/YUV/"
    rf = RandomForest(random_state=42, oob_score=True, verbose=0, n_estimators=440, max_depth=30, max_features='sqrt', min_samples_leaf=2) if rf is None else rf
    train_model_no_search(args, rf, preprocess=YUV_single_patch_per_latent, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)
    test_process(args, preprocess=YUV_single_patch_per_latent)

    

def exp_flatten_RF_no_search(args,rf=None, X_raw=None, X_hat_raw=None, labels=None):
    args.models_save_dir = "/data/lesc/users/rustichini/thesis/models_saved/FLATTEN/"
    train_RF_no_search(args, rf, preprocess=flatten_latents, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)
    test_process(args, preprocess=flatten_latents)

def exp_learning_curve_Y_RF(args):
    rf = RandomForest(random_state=42, oob_score=True, verbose=0, n_estimators=670, max_depth=42, max_features='sqrt', min_samples_leaf=2)
    preprocess = Y_single_patch_per_latent

    print(f"\n {10*'*'} Training Learning Curve {10*'*'}")

    learn_curve_process(args, rf, preprocess)
    learn_curve_process(args, rf, preprocess)

def all_exp_RF_flatten(args):
    rf = RandomForest(random_state=42, oob_score=True, verbose=0, n_estimators=470, max_depth=30, max_features='sqrt', min_samples_leaf=2)
    for bpp in [600, 1200]:
        args.set_target_bpp = bpp
        save_latent_path = os.path.join(args.bin_path, str(args.set_target_bpp)+"_bpp", "latent") if args.save_latents else None
        X_raw, X_hat_raw, labels = prepare_dataset(args, args.train_csv, save_latent_path)
        exp_flatten_RF_no_search(args, rf, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)

def all_exp_RF_single(args):
    rf = RandomForest(random_state=42, oob_score=True, verbose=0, n_estimators=470, max_depth=30, max_features='sqrt', min_samples_leaf=2)
    for bpp in [600, 1200]:
        args.set_target_bpp = bpp
        save_latent_path = os.path.join(args.bin_path, str(args.set_target_bpp)+"_bpp", "latent") if args.save_latents else None
        X_raw, X_hat_raw, labels = prepare_dataset(args, args.train_csv, save_latent_path)
        exp_single_Y_RF_no_search(args, rf, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)
        exp_single_YUV_RF_no_search(args, rf, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)
        #exp_flatten_RF_no_search(args, rf, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)


def all_exp_RF_multiple(args):
    rf = RandomForest(random_state=42, oob_score=True, verbose=0, n_estimators=463, max_depth=30, max_features='sqrt', min_samples_leaf=2)
    
    for bpp in [600, 1200]:
        args.set_target_bpp = bpp
        save_latent_path = os.path.join(args.bin_path, str(args.set_target_bpp)+"_bpp", "latent") if args.save_latents else None
        X_raw, X_hat_raw, labels = prepare_dataset(args, args.train_csv, save_latent_path)
        
        exp_multiple_Y_RF_no_search(args, rf, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)
        exp_multiple_YUV_RF_no_search(args, rf, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)

def exp_RF_default(args):
    rf = RandomForest(random_state=42)
    for bpp in [600, 1200]:
        args.set_target_bpp = bpp
        save_latent_path = os.path.join(args.bin_path, str(args.set_target_bpp)+"_bpp", "latent") if args.save_latents else None
        X_raw, X_hat_raw, labels = prepare_dataset(args, args.train_csv, save_latent_path)
        exp_single_Y_RF_no_search(args, rf, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)
        exp_single_YUV_RF_no_search(args, rf, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)
        #exp_multiple_Y_RF_no_search(args, rf, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)
        #exp_multiple_YUV_RF_no_search(args, rf, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)
        #exp_flatten_RF_no_search(args, rf, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)

if __name__ == "__main__":
    args = setup_parser()
    print(f"{10*'*'} EXPERIMENTS ON RF {10*'*'}")

    """
    # Train
    rf = RandomForest(random_state=42, oob_score=True, verbose=0)
    preprocess = single_patch_per_latent
    #train_grid_search(args, estimator=rf, preprocess=preprocess)

    args.num_samples = 20000
    train_RF_random(args, preprocess=preprocess)
    #train_model_no_search(args, RandomForest(random_state=42, n_estimators=583,max_depth=30, max_features='sqrt', min_samples_leaf=2), preprocess=preprocess)
    args.num_samples_test = 1000
    #args.model_file_y = "/data/lesc/users/rustichini/thesis/models_saved/SINGLE_PATCH/YUV/NEW/600_bpp/10000_samples/y/RF_583estimators.joblib"
    #args.model_file_y_hat = "/data/lesc/users/rustichini/thesis/models_saved/SINGLE_PATCH/YUV/NEW/600_bpp/10000_samples/y_hat/RF_583estimators.joblib"
    test_process(args, preprocess=preprocess)
    """

    
    
    args.save_latents = True

    args.num_samples = 10000
    args.num_samples_test = 1000
    #all_exp_RF_single(args)
    #args.set_target_bpp = 600
    rf = RandomForest(max_depth=25, max_features='sqrt', min_samples_leaf=4,
                       n_estimators=463, oob_score=True, random_state=42)
    #exp_single_Y_RF_random_search(args)
    #exp_multiple_Y_RF_random_search(args)
    #exp_multiple_YUV_RF_random_search(args)
    all_exp_RF_multiple(args)
    #all_exp_RF_flatten(args)
