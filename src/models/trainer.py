import copy
import sys
sys.path.append('../')
from config import setup_parser
import os
import numpy as np
import time
from data.dataset_manager import  prepare_dataset
from data.preprocessing import flatten_latents_YUV, create_patches_dataset
from manager import ModelManager, GridSearch, RandomForest, RandomizedSearch
from sklearn.ensemble import RandomForestClassifier
from common import format_time


def training(args, model, X_train, labels, preprocess, target: str):
    model_manager = ModelManager(model, preprocess)
    save_model_path = os.path.join(args.models_save_dir,str(args.set_target_bpp)+"_bpp",str(len(X_train)) + "_samples")

    model_path_file = os.path.join(save_model_path, target)

    model_manager.train_model(X_train, labels, model_path_file)

    if target=='y':
        args.model_file_y = os.path.join(model_path_file, model.name + '.joblib' )
    elif target=='y_hat':
        args.model_file_y_hat = os.path.join(model_path_file, model.name + '.joblib' )


def learning_curve(args, model, X_train, labels, preprocess, target: str):
    model_manager = ModelManager(model, preprocess)
    save_model_path = os.path.join(args.models_save_dir,"Learning_curve",str(args.set_target_bpp)+"_bpp", target)
    os.makedirs(save_model_path, exist_ok=True)
    model_manager.learning_curve(X_train, labels, save_model_path)
def learn_curve(args, model,X_train,labels, preprocess, target: str):
    if target=='y':
        learning_curve(args, model, X_train, labels, preprocess, target=target)
    elif target=='y_hat':
        learning_curve(args, model, X_train, labels, preprocess, target=target)
def learn_curve_process(args, model, preprocess):
    print(f"\n {10*'*'} Learning Curve {10*'*'}")
    save_latent_path = os.path.join(args.bin_path,str(args.set_target_bpp)+"_bpp","latent") if args.save_latents else None
    model_y = copy.deepcopy(model)
    model_y_hat = copy.deepcopy(model)
    X_raw, X_hat_raw, labels = prepare_dataset(args, args.train_csv, save_latent_path)

    learn_curve(args, model_y, X_raw, labels, preprocess, target='y')
    del X_raw
    learn_curve(args, model_y_hat, X_hat_raw, labels, preprocess, target='y_hat')

    

def train_process(args, model, preprocess, X_raw=None, X_hat_raw=None, labels=None):
    print(f"\n {10*'*'} Training {10*'*'}")

#    print("Model to train: " + model.best_estimator if hasattr(model, 'best_estimator') else model)

    model_y = copy.deepcopy(model)
    model_y_hat = copy.deepcopy(model)
    
    if X_raw is None or X_hat_raw is None or labels is None:
        save_latent_path = os.path.join(args.bin_path,str(args.set_target_bpp)+"_bpp","latent") if args.save_latents else None
        X_raw, X_hat_raw, labels = prepare_dataset(args, args.train_csv, save_latent_path)

    print("\nTraining on target y")
    start_time = time.time()
    training(args, model_y, X_raw, labels, preprocess, target='y')
    end_time = time.time()
    print_time(start_time, end_time)

    #del X_raw FIXME

    print("\nTraining on target y_hat")
    start_time = time.time()
    training(args, model_y_hat, X_hat_raw, labels, preprocess, target='y_hat')
    end_time = time.time()
    print_time(start_time, end_time)

def train_grid_search(args, estimator,param_grid, preprocess, X_raw=None, X_hat_raw=None, labels=None):
    """
    Train a model using Grid Search for hyperparameter tuning.
    """
    model_to_train =GridSearch(
        estimator=estimator, 
        name=args.model_name, 
        params=param_grid)

    train_process(args, model_to_train, preprocess=preprocess, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)


def train_random_search(args, estimator,param_distributions, preprocess, n_iter = 100, X_raw=None, X_hat_raw=None, labels=None):
    
    random_search = RandomizedSearch(
        estimator=estimator, 
        name=args.model_name,
        params=param_distributions,
        n_iter=n_iter,
    )
    train_process(args, random_search, preprocess, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)
    """
    search_results = random_search
    best_model = search_results.best_estimator_
    print("\nBest model found:", best_model)
    print("\nBest hyperparameters:", search_results.best_params_)
    print("\nBest score during search:", search_results.best_score_)
    """
    


def train_model_no_search(args, model, preprocess, X_raw=None, X_hat_raw=None, labels=None):
    train_process(args, model, preprocess, X_raw=X_raw, X_hat_raw=X_hat_raw, labels=labels)


def print_time(start_time, end_time):
    train_time = format_time(end_time - start_time)
    print(f"Training time: {train_time}")
    
if __name__ == "__main__":
    pass

