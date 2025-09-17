import sys
from manager import ModelManager
sys.path.append('../')
from config import setup_parser
from common import load_on_RAM
from data.dataset_manager import prepare_dataset
from data.preprocessing import create_patches_dataset, single_patch_per_latent

import os


def testing(model, X, y, preprocess, target:str):
    """
    Calls the test method of ModelManager
    """
    print("Test on target " + target)
    model_manager = ModelManager(model, preprocess)
    model_manager.test_model(X,y)


def test_process(arg, preprocess):
    args.num_samples = args.num_samples_test if args.num_samples_test is not None else args.num_samples
    save_latent_path = os.path.join(args.bin_path,str(args.set_target_bpp)+"_bpp","test","latent")

    X_raw, X_hat_raw, labels = prepare_dataset(args, args.test_csv, save_latent_path)

    test_single_target(X_raw, labels, args.model_file_y, preprocess, target='y')
    del X_raw
    test_single_target(X_hat_raw, labels, args.model_file_y_hat,preprocess, target='y_hat')

def test_single_target(X,y, model_path,preprocess, target):
    model_file_path = model_path
    model = load_on_RAM(model_file_path)
    print("Loaded model from " + model_file_path)
    if target=='y':
        testing(model, X, y, preprocess, target=target)
    elif target=='y_hat':
        testing(model, X, y, preprocess, target=target)


if __name__ == "__main__":
    args = setup_parser()  

    args.num_samples_test = 4000

#    args.test_csv = ""
    args.model_file_y = "/data/lesc/users/rustichini/thesis/models_saved/YUV/1200_bpp/30000_samples/y/grid_search.joblib"
    args.model_file_y_hat = "/data/lesc/users/rustichini/thesis/models_saved/YUV/1200_bpp/30000_samples/y_hat/grid_search.joblib"

    test_process(args, preprocess=single_patch_per_latent)
    model = load_on_RAM(args.model_file_y)
    print("Loaded model from " + args.model_file_y)
    #print(model)
    print(model.best_estimator_)
    print(model.best_estimator_.oob_score_)

