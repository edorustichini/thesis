from random import shuffle

import torch
from utils import save
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from utils import save

from parser import setup_parser
from coder import CoderManager
from dataset import DatasetManager
import pandas as pd
from main import clean_dataframe
import os
import numpy as np
from sklearn.utils import shuffle


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
        
def flatten_latents(latents):
    flat_set = []
    for latent in latents:
        flat = torch.cat([torch.flatten(latents['model_y']), torch.flatten(latents['model_uy'])]).cpu().numpy()
        flat_set.append(flat)
    return flat_set
    
    
    
def prepare_dataset(args):
    # Coder setup
    coder_manager = CoderManager(args)
    
    # Dataset setup
    dataset_manager = DatasetManager(coder_manager.coder)
    
    # -- Dataset creation --
    df_train = pd.read_csv(args.train_csv)
    #df_train = dataset_manager.clean_dataframe(df_train)
    
    if args.num_samples is not None:
        df_train = dataset_manager.sample_subset(df_train, args.num_samples, args.random_sample)
    
    
    save_latent_path = os.path.join(args.bin_path,str(args.set_target_bpp)+"_bpp","latent") if args.save else None
    
    #extract latent
    X_raw, X_hat_raw, labels = dataset_manager.build_latent_dataset(
        df_train,
        args.img_dir,
        save_latent_path)
    return X_raw, X_hat_raw, labels

def training(X_raw, labels, target: str):
    # pre-process latents
    y_train = np.array(labels)
    X_train = np.array(flatten_latents(X_raw))
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

    trainer.train_model(X_train, labels, os.path.join(save_model_path, target))


def train_process(args):
    X_raw, X_hat_raw, labels = prepare_dataset(args)
    training(X_raw, labels, target='y')
    del X_raw
    training(X_hat_raw, labels, target='y_hat')


if __name__ == "__main__":
    # Argument parsing
    args = setup_parser()
    train_process(args)
   