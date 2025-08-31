from parser import setup_parser
from main import prepare_dataset
from common import load_on_RAM, save
from dataset import flatten_latents
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from trainer import ModelManager, GridSearch, RandomForest


def testing(model, X, y, preprocess, target:str):
    print("Test on target " + target)
    
    model_manager = ModelManager(model, preprocess)
    model_manager.test_model(X,y)


def test_process(args):
    model_file_path = args.model_file # TODO: args.model_test
    model = load_on_RAM(model_file_path)
    print("Loaded model from " + model_file_path)

    print(model)

    save_latent_path = os.path.join(args.bin_path,str(args.set_target_bpp)+"_bpp","test","latent")
    #print("Preparing test set")
    X_raw, _, labels = prepare_dataset(args, args.test_csv)
    testing(model, X_raw, labels, flatten_latents, target='y')

if __name__ == "__main__":
    args = setup_parser()
    model_file_path = args.model_file # TODO: args.model_test
    model = load_on_RAM(model_file_path)
    grid = model
    best_rf = grid.best_estimator_
    print("OOB score for best model found by GridSearch", best_rf.oob_score_)
    print("Best parameters found:", grid.best_params_)
    test_process(args)