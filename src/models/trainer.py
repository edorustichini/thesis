import sys
import os
import numpy as np
import time



from manager import ModelManager, GridSearch, RandomForest, RandomizedSearch, SVM
from pipeline import Pipeline


def training(args, model, X_train, labels, preprocess, target: str):
    model_manager = ModelManager(model, preprocess)
    save_model_path = os.path.join(args.models_save_dir,str(args.set_target_bpp)+"_bpp",str(len(X_train)) + "_samples")

    model_path_file = os.path.join(save_model_path, target)

    model_manager.train_model(X_train, labels, model_path_file)

    if target=='y':
        args.model_file_y = os.path.join(model_path_file, model.name + '.joblib' )
    elif target=='y_hat':
        args.model_file_y_hat = os.path.join(model_path_file, model.name + '.joblib' )

def train_grid_search(args, estimator,param_grid, preprocess):
    """
    Train a model using Grid Search for hyperparameter tuning.
    """
    grid_search =GridSearch(
        estimator=estimator, 
        name=args.model_name, 
        params=param_grid)
    
    exp = Pipeline(
        model = grid_search,
        train_preprocess=preprocess)
    exp.run_training_only()
    search_results = grid_search
    best_model = search_results.best_estimator_
    print("\nBest model found:", best_model)
    print("\nBest hyperparameters:", search_results.best_params_)
    print("\nBest score during search:", search_results.best_score_)
    
def train_random_search(args, estimator,param_distributions, preprocess):
    
    random_search = RandomizedSearch(
        estimator=estimator, 
        name=args.model_name,
        params=param_distributions,
        n_iter=50
    )
    exp = Pipeline(
        model = random_search,
        train_preprocess=preprocess
    )
    exp.run_training_only()
    search_results = random_search
    best_model = search_results.best_estimator_
    print("\nBest model found:", best_model)
    print("\nBest hyperparameters:", search_results.best_params_)
    print("\nBest score during search:", search_results.best_score_)

def train_model_no_search(args, model, preprocess):
    pass
    #train_process(args, model, preprocess)
    fitted_model = model
    print("\nOOB score for model:", fitted_model.oob_score_) if hasattr(fitted_model, 'oob_score_') else "N/A"
    print("\nFinal model:", fitted_model)

 
if __name__ == "__main__":
    pass

