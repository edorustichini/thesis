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


class ModelTrainer:
    def __init__(self, model):
        """
        model must have "fit" function, defined like scikit-learn fit, and a attribute for the name
        """
        self.model = model
    
    def train_model(self, X, y, save_path):
        X, y = shuffle(X, y, random_state=42)
        print("Training started")
        self.model.fit(X, y)
        print("Training finished")
        
        if save_path is not None:
            self.save_model(save_path)

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
    def __init__(self, name:str = None, **params):
        super().__init__(**params)
        if name is None:
            name = f"grid_search"
        self.name = name

def training(X_train, labels, target: str):
    # pre-process latents
    y_train = np.array(labels)
    X_train = np.array(flatten_latents(X_train))
    grid_params = {
        'n_estimators': [50, 75, 100],
        'max_features': [0.05, 0.1, 'sqrt'],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [10, 20],
        'min_samples_leaf': [2, 5],
        'bootstrap': [True],
    }
    model = GridSearch(name=None, **grid_params)
    trainer = ModelTrainer(model)
    save_model_path = os.path.join(args.models_save_dir,str(args.set_target_bpp)+"_bpp",str(len(X_train)) + "_samples")

    trainer.train_model(X_train, y_train, os.path.join(save_model_path, target))


def train_process(args):
    X_raw, X_hat_raw, labels = prepare_dataset(args)
    training(X_raw, labels, target='y')
    del X_raw
    training(X_hat_raw, labels, target='y_hat')


if __name__ == "__main__":
    # Argument parsing
    """
    Example of use: 
        python trainer.py ../../input_imgs/ ../../JPEGAI_output/ --set_target_bpp 600 --num_samples 30000 --gpu 0
    """
    args = setup_parser()
    train_process(args)
   
