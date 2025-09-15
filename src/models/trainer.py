import sys
sys.path.append('../')
from config import setup_parser
import os
import numpy as np
import time
from data.dataset_manager import  prepare_dataset
from data.preprocessing import flatten_latents, create_patches_dataset
from manager import ModelManager, GridSearch, RandomForest, RandomizedSearch
from sklearn.ensemble import RandomForestClassifier
from common import format_time


def training(args, model, X_train, labels, preprocess, target: str):
    model_manager = ModelManager(model, preprocess)
    save_model_path = os.path.join(args.models_save_dir,str(args.set_target_bpp)+"_bpp",str(len(X_train)) + "_samples")

    model_manager.train_model(X_train, labels, os.path.join(save_model_path, target))


def train_process(args, model, preprocess):
    save_latent_path = os.path.join(args.bin_path,str(args.set_target_bpp)+"_bpp","latent") if args.save_latents else None
    
    X_raw, X_hat_raw, labels = prepare_dataset(args, args.train_csv, save_latent_path)

    print("\nTraining on target y")
    start_time = time.time()
    training(args, model, X_raw, labels, preprocess, target='y')
    end_time = time.time()
    train_time = format_time(end_time - start_time)
    print(f"Training time: {train_time}")
    del X_raw

    print("\nTraining on target y_hat")
    start_time = time.time()
    training(args, model, X_hat_raw, labels, preprocess, target='y_hat')
    end_time = time.time()
    train_time = format_time(end_time - start_time)
    print(f"Training time: {train_time}")

def train_grid_search(args, estimator,param_grid, preprocess):
    """
    Train a model using Grid Search for hyperparameter tuning.
    """
    model_to_train =GridSearch(
        estimator=estimator, 
        name=args.model_name, 
        params=param_grid)

    train_process(args, model_to_train, preprocess=preprocess)
    
    grid = model_to_train
    best_rf = grid.best_estimator_
    print("\nOOB score for best model found by GridSearch", best_rf.oob_score_)
    print("\nBest parameters found:", grid.best_params_)
    print("\nBest cross-validation score:", grid.best_score_)
    print("\nFinal model: ", best_rf)

def train_random_search(args, estimator,param_distributions, preprocess):
    
    random_search = RandomizedSearch(
        estimator=estimator, 
        name=args.model_name,
        params=param_distributions,
        n_iter=50
    )
    train_process(args, random_search, preprocess)
    search_results = random_search
    best_rf = search_results.best_estimator_
    print("\nOOB score for best model found:", best_rf.oob_score_) if hasattr(best_rf, 'oob_score_') else "N/A"
    print("\nBest parameters found:", search_results.best_params_)
    print("\nBest cross-validation score:", search_results.best_score_)
    print("\nFinal model:", best_rf)

def train_model_no_search(args, model, preprocess):
    train_process(args, model, preprocess)
    fitted_model = model
    print("\nOOB score for model:", fitted_model.oob_score_) if hasattr(fitted_model, 'oob_score_') else "N/A"
    print("\nFinal model:", fitted_model)

if __name__ == "__main__":
    pass

