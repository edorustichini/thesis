from manager import GridSearch, RandomizedSearch
from sklearn.ensemble import RandomForestClassifier
from trainer import train_grid_search, train_random_search, train_model_no_search

import sys
sys.path.append('../')
from config import setup_parser
from data.preprocessing import flatten_latents, create_patches_dataset

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
    'n_estimators': stats.randint(100, 501), 
    'max_depth': stats.randint(5, 31),
    
    'min_samples_split': stats.randint(2, 21),  
    'min_samples_leaf': stats.randint(1, 16),  
    #'min_samples_split': stats.uniform(0.01, 0.19),  # da 1% a 20% dei samples
    #'min_samples_leaf': stats.uniform(0.01, 0.09),   # da 1% a 10% dei samples
        
    'max_features': ['sqrt'],
    'bootstrap': [True]
    }

    param_distributions = [

    { # Small trees
        'n_estimators': stats.randint(50, 201),
        'max_depth': stats.randint(5, 16),
        'min_samples_leaf': stats.randint(10, 21),
        'max_features': ['sqrt'],
        'bootstrap': [True]
    },
    { # Large trees
        'n_estimators': stats.randint(200, 501),
        'max_depth': stats.randint(20, 51),
        'min_samples_leaf': stats.randint(1, 6),
        'max_features': ['sqrt'],
        'bootstrap': [True]
    },
    { # No bootstrapping
        'n_estimators': stats.randint(100, 301),
        'max_depth': stats.randint(15, 31),
        'min_samples_leaf': stats.randint(1, 11),
        'bootstrap': [False],
        'max_features': [None],
        'min_samples_split': stats.randint(2, 11)
    }
    ]
    
    rf = RandomForestClassifier(random_state=42, oob_score=True, verbose=0)
    train_random_search(args, rf, param_distributions, preprocess)


if __name__ == "__main__":
    args = setup_parser()
    
    # Train
    #train_grid_search(args, preprocess=create_patches_dataset)
    #train_random_search(args, preprocess=create_patches_dataset)
    rf = RandomForestClassifier(random_state=42, oob_score=True, verbose=0)
    preprocess = create_patches_dataset
    train_model_no_search(args, model=rf, preprocess=preprocess)

    