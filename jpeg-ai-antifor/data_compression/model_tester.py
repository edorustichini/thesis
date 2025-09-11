from parser import setup_parser
from main import prepare_dataset
from common import load_on_RAM, save
from dataset import flatten_latents, create_patches_dataset
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

    
    print(model.best_estimator_)
    print("OOB score for best model found by GridSearch", model.best_estimator_.oob_score_)
    

    save_latent_path = os.path.join(args.bin_path,str(args.set_target_bpp)+"_bpp","test","latent")
    print("Preparing test set")
    X_raw, _, labels = prepare_dataset(args, args.test_csv, save_latent_path)
    testing(model, X_raw, labels, create_patches_dataset, target='y')



def testing_with_majority_vote(model, X, y, channels_per_image=160):
    print(f"Test with majority vote - {channels_per_image} channels per image")
    
    model_manager = ModelManager(model, None)  # No preprocessing needed
    predictions = model_manager.test_model_with_majority_vote(X, y, channels_per_image)
    return predictions

def test_process_majority_vote(args):
    model_file_path = args.model_file
    model = load_on_RAM(model_file_path)
    print("Loaded model from " + model_file_path)
    print(model)

    save_latent_path = os.path.join(args.bin_path, str(args.set_target_bpp)+"_bpp", "test", "latent")
    
    X_raw, _, labels = prepare_dataset(args, args.test_csv, save_latent_path)
    
    channels_per_image = 160
    X_grouped, labels = channels_to_tensor_for_testing(X_raw, labels, channels_per_image)
    
    # Test con voto a maggioranza
    majority_predictions = testing_with_majority_vote(model, X_grouped, labels, channels_per_image)
    



if __name__ == "__main__":
    args = setup_parser()
    model_file_path = args.model_file # TODO: args.model_test
    model = load_on_RAM(model_file_path)
    #best_rf = grid.best_estimator_
    #print("OOB score for best model found by GridSearch", best_rf.oob_score_)
    #print("Best parameters found:", grid.best_params_)
    test_process(args)
    #test_process_majority_vote(args)
