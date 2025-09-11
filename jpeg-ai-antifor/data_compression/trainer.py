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
from dataset import flatten_latents, create_patches_dataset


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

    def test_model_with_majority_vote(self, X_grouped, y, channels_per_image=10):
        """
        Test con voto a maggioranza
        
        Args:
            X_grouped: lista di liste, ogni sottolista contiene i canali di un'immagine
            y: labels vere (una per immagine)
            channels_per_image: numero di canali per immagine
        """
        predictions_per_image = []
        
        print(f"Testing {len(X_grouped)} images with {channels_per_image} channels each...")
        
        for i, image_channels in enumerate(X_grouped):
            # Predici per ogni canale dell'immagine
            channel_predictions = []
            
            for channel in image_channels:
                channel_reshaped = np.array(channel).reshape(1, -1)
                pred = self.model.predict(channel_reshaped)[0]
                channel_predictions.append(pred)
            
            # Voto a maggioranza
            majority_vote = max(set(channel_predictions), key=channel_predictions.count)
            predictions_per_image.append(majority_vote)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(X_grouped)} images")
        
        # Calcola metriche
        y = np.array(y)
        predictions_per_image = np.array(predictions_per_image)
        
        print("\n=== MAJORITY VOTE RESULTS ===")
        print(classification_report(y, predictions_per_image))
        print("Accuracy score:", accuracy_score(y, predictions_per_image))
        print("Confusion Matrix:")
        print(confusion_matrix(y, predictions_per_image))
        
        return predictions_per_image
        
    
    def preprocess_sets(self, X, y):
        # pre-process latents
        X, y = self.preprocess(X, y)

        y = np.array(y)
        X = np.array(X)
        return X,y

    def save_model(self, save_path):
        print(self.model.name + "will be saved into "+ save_path + " as "+ self.model.name)
        save(self.model, save_path, self.model.name)
    
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

