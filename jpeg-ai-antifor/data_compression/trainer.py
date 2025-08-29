from random import shuffle

import torch
from common import save
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from parser import setup_parser
import pandas as pd
import os
import numpy as np
from sklearn.utils import shuffle
from main import prepare_dataset
from dataset import flatten_latents


class ModelManager:
    def __init__(self, model, preprocess):
        """
        model must have "fit" function, defined like scikit-learn fit, and a attribute for the name
        """
        self.model = model
        self.preprocess = preprocess
    
    def train_model(self, X, y, save_path):
        X, y = self.preprocess_sets(X,y)
        X, y = shuffle(X, y, random_state=42)
        print("Training started")
        self.model.fit(X, y)
        print("Training finished")
        
        if save_path is not None:
            self.save_model(save_path)
        
    
    def preprocess_sets(self, X, y):
        # pre-process latents
        X, y = self.preprocess(X, y)

        y = np.array(y)
        X = np.array(X)
        return X,y

    def save_model(self, save_path):
        print(self.model.name + "will be saved into"+ save_path)
        save(self.model, save_path, self.model.name )
    
class RandomForest(RandomForestClassifier):
    def __init__(self, name:str = None, **params):
        super().__init__(**params)
        if name is None:
            name = f"RF_{params['n_estimators']}trees"
        self.name = name

class GridSearch(GridSearchCV):
    def __init__(self, estimator, name,params):
        super().__init__(estimator=estimator, param_grid=params)
        if name is None:
            name = f"grid_search"
        self.name = name

def training(model, X_train, labels, preprocess, target: str):
    
    trainer = ModelManager(model, preprocess)
    save_model_path = os.path.join(args.models_save_dir,str(args.set_target_bpp)+"_bpp",str(len(X_train)) + "_samples")

    trainer.train_model(X_train, labels, os.path.join(save_model_path, target))


def train_process(args, model):
    X_raw, X_hat_raw, labels = prepare_dataset(args)
    training(model, X_raw, labels,flatten_latents, target='y')
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
        'n_estimators': [100, 120, 150],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [3, 7, 10,20],
        'min_samples_split': [10, 20],
        'min_samples_leaf': [1,2,4],
        'bootstrap': [True],
    }
    model_to_train = GridSearch(estimator=RandomForestClassifier(random_state=42, oob_score=True), name=None, params=param_grid)
    
    train_process(args, model_to_train)

    grid = model_to_train
    best_rf = grid.best_estimator_
    print("OOB score for best model found by GridSearch", best_rf.oob_score_)
    print("Best parameters found:", grid.best_params_)
