from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import sys
sys.path.append('../')
from config import setup_parser
import os
import numpy as np
from data.dataset_manager import  prepare_dataset
from data.preprocessing import flatten_latents, create_patches_dataset
from manager import ModelManager
    
class RandomForest(RandomForestClassifier):
    def __init__(self, name=None, **params):
        super().__init__(**params)
        if name is None:
            name = f"RF_{params['n_estimators']}estimators"
        self.name = name  

class GridSearch(GridSearchCV):
    def __init__(self, estimator, name,params):
        super().__init__(estimator=estimator, param_grid=params, n_jobs=-1)
        if name is None:
            name = f"grid_search"
        self.name = name

def training(model, X_train, labels, preprocess, target: str):
    
    model_manager = ModelManager(model, preprocess)
    save_model_path = os.path.join(args.models_save_dir,str(args.set_target_bpp)+"_bpp",str(len(X_train)) + "_samples")

    model_manager.train_model(X_train, labels, os.path.join(save_model_path, target))


def train_process(args, model, preprocess=flatten_latents):
    save_latent_path = os.path.join(args.bin_path,str(args.set_target_bpp)+"_bpp","latent") if args.save else None

    X_raw, X_hat_raw, labels = prepare_dataset(args, args.train_csv, save_latent_path)
    training(model, X_raw, labels,preprocess, target='y')
    # TODO: fai anche y_hat
    #del X_raw
    #training(model,X_hat_raw, labels,flatten_latents, target='y_hat')



if __name__ == "__main__":
    """
    Example of use: 
        python trainer.py ../../input_imgs/ ../../JPEGAI_output/ --set_target_bpp 600 --num_samples 30000 --gpu 0
    """
    # Argument parsing
    args = setup_parser()

    # Setup model for training
    # param_grid = {
    #    'n_estimators': [20, 50, 75, 100],
    #    'max_depth': [3, 5,7, 10,15],
    #    'max_features' : ['sqrt', 'log2'],
    #    'min_samples_split': [10, 20],
    #    'min_samples_leaf': [1,2,4],
    #    'bootstrap': [True],
    #}
    param_grid = {
        'n_estimators': [250, 350, 450],
        'max_depth': [15, 20, 30],
        'min_samples_leaf': [ 8, 10, 12, 15],
    }
    rf = RandomForestClassifier(random_state=42, oob_score=True, verbose=0)
    model_to_train =GridSearch(estimator=rf, name=None, params=param_grid)

    train_process(args, model_to_train, preprocess=create_patches_dataset)

    grid = model_to_train
    best_rf = grid.best_estimator_
    print("\nOOB score for best model found by GridSearch", best_rf.oob_score_)
    print("\nBest parameters found:", grid.best_params_)
    print("\nFinal model: ", best_rf)

