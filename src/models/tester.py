import sys
from manager import ModelManager
sys.path.append('../')
from config import setup_parser
from common import load_on_RAM
from data.dataset_manager import prepare_dataset
from data.preprocessing import create_patches_dataset

import os


def testing(model, X, y, preprocess, target:str):
    """
    Calls the test method of ModelManager
    """
    print("Test on target " + target)
    model_manager = ModelManager(model, preprocess)
    model_manager.test_model(X,y) 


def test_process(args,model_file_path_y: str = None, model_file_path_y_hat: str = None):
    args.num_samples = args.num_samples_test if args.num_samples_test is not None else args.num_samples
    save_latent_path = os.path.join(args.bin_path,str(args.set_target_bpp)+"_bpp","test","latent")

    X_raw, X_hat_raw, labels = prepare_dataset(args, args.test_csv, save_latent_path)

    test_single_target(args, X_raw, labels, model_file_path_y, target='y')
    del X_raw
    test_single_target(args, X_hat_raw, labels, model_file_path_y_hat, target='y_hat')

def test_single_target(args,X,y, model_path, target):
    model_file_path = model_path
    model = load_on_RAM(model_file_path)
    print("Loaded model from " + model_file_path)
    if target=='y':
        testing(model, X, y, create_patches_dataset, target=target)
    elif target=='y_hat':
        testing(model, X, y, create_patches_dataset, target=target)


if __name__ == "__main__":
    args = setup_parser()    
    test_process(args)
    
