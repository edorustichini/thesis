from manager import GridSearch, RandomizedSearch
from tester import test_process
from sklearn.ensemble import RandomForestClassifier
from trainer import train_grid_search, train_random_search, train_model_no_search

import sys
sys.path.append('../')
from config import setup_parser
from pipeline import Pipeline
from data.preprocessing import flatten_latents, single_patch_per_latent, multiple_patches_per_latent

def train_RF_grid_search(args, preprocess):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    rf = RandomForestClassifier(random_state=42, oob_score=True, verbose=0)
    train_grid_search(args, rf, param_grid, preprocess)

def train_RF_random_search(args, preprocess):
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
        #single_param_distributions, # Small trees
        # Large trees
        {
            'n_estimators': stats.randint(500, 800),
            'max_depth': stats.randint(25, 51),
            'min_samples_leaf': [2],
            'max_features': ['sqrt'],
            'bootstrap': [True]
        }
    ]
    estimator = RandomForestClassifier(random_state=42, oob_score=True, verbose=0)
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

    #train_random_search(args, rf, param_distributions, preprocess)

if __name__ == "__main__":
    args = setup_parser()
    
    # Train
    rf = RandomForestClassifier(random_state=42, oob_score=True, verbose=0)
    preprocess = single_patch_per_latent
    
    train_RF_random_search(args, preprocess=preprocess)

    