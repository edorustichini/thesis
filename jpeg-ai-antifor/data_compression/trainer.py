from random import shuffle

import torch
from common import save
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from parser import setup_parser
import pandas as pd
import os
import numpy as np
from sklearn.utils import shuffle
from main import prepare_dataset
from dataset import flatten_latents, channels_to_tensor


class ModelManager:
    # TODO: spostare questa classe in un altro file
    def __init__(self, model, preprocess):
        """
        model must have "fit" function, defined like scikit-learn fit, and a attribute for the name
        """
        self.model = model
        self.preprocess = preprocess
    
    def train_model(self, X, y, save_path):
        X, y = self.preprocess_sets(X,y)
        X, y = shuffle(X, y, random_state=42)

        print("\n(num_samples, num_features)")
        print(f"Train dataset : {X.shape}\n")
        print("Training started")
        self.model.fit(X, y)
        print("Training finished")
        
        self.save_model(save_path)
    
    def test_model(self, X, y):
        X, y = self.preprocess_sets(X,y)
        X, y= shuffle(X, y, random_state=42)

        y_pred = self.model.predict(X)

        print(classification_report(y, y_pred))

        print("Accuray score : " + str(accuracy_score(y, y_pred)))
        
    
    def preprocess_sets(self, X, y):
        # pre-process latents
        X, y = self.preprocess(X, y)

        y = np.array(y)
        X = np.array(X)
        return X,y

    def save_model(self, save_path):
        print(self.model.name + "will be saved into "+ save_path + " as "+ self.model.name)
        save(self.model, save_path, self.model.name )
    
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
        'n_estimators': [75, 100, 125],
        'max_depth': [20,40,60],
        'min_samples_leaf': [1,2,4],
        'bootstrap': [True],
    }
    rf = RandomForestClassifier(random_state=42, oob_score=True, verbose=1)
    model_to_train =GridSearch(estimator=rf, name=None, params=param_grid)
    
    train_process(args, model_to_train, preprocess=channels_to_tensor)

    """
    grid = model_to_train
    best_rf = grid.best_estimator_
    print("OOB score for best model found by GridSearch", best_rf.oob_score_)
    print("Best parameters found:", grid.best_params_)
"""
