from manager import GridSearch, RandomizedSearch
from tester import test_process
from sklearn.ensemble import RandomForestClassifier
from trainer import train_grid_search, train_random_search, train_model_no_search

import sys
sys.path.append('../')
from config import setup_parser
from data.preprocessing import flatten_latents, single_patch_per_latent, multiple_patches_per_latent

def train_RF_grid(args, preprocess):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    rf = RandomForestClassifier(random_state=42, oob_score=True, verbose=0)
    train_grid_search(args, rf, param_grid, preprocess)

def train_RF_random(args, preprocess):
    import scipy.stats as stats
    single_param_distributions = {
    'n_estimators': stats.randint(100, 300), 
    'max_depth': stats.randint(5, 21), 
    'min_samples_split': stats.uniform(0.01, 0.19),  # da 1% a 20% dei samples
    'min_samples_leaf': stats.uniform(0.01, 0.09),   # da 1% a 10% dei samples
        
    'max_features': ['sqrt'],
    'bootstrap': [True]
    }

    param_distributions = [
    single_param_distributions, # Small trees
    { # Large trees
        'n_estimators': stats.randint(300, 800),
        'max_depth': stats.randint(10, 51),
        'min_samples_leaf': stats.randint(2, 15),
        'max_features': ['sqrt'],
        'bootstrap': [True]
    }
    ]
    
    rf = RandomForestClassifier(random_state=42, oob_score=True, verbose=0)
    train_random_search(args, rf, param_distributions, preprocess)


if __name__ == "__main__":
    args = setup_parser()
    
    # Train
    rf = RandomForestClassifier(random_state=42, oob_score=True, verbose=0)
    preprocess = single_patch_per_latent
    #train_grid_search(args, estimator=rf, preprocess=preprocess)

    train_RF_random(args, preprocess=preprocess)

    #test_process(args, model_file_path_y=args.model_file_y, model_file_path_y_hat=args.model_file_y_hat)
    
    

    #train_model_no_search(args, model=rf, preprocess=preprocess)