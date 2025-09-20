import sys
from manager import ModelManager
sys.path.append('../')
from config import setup_parser
from common import load_on_RAM
from data.dataset_manager import prepare_dataset
from data.preprocessing import Y_all_patches_per_latent, YUV_all_patches_per_latent, YUV_multiple_patches_per_latent, create_patches_dataset, YUV_single_patch_per_latent

import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def testing(model, X, y, preprocess, target:str):
    """
    Calls the test method of ModelManager
    """
    import numpy as np
    _,_,H,W = X[0]['model_y'].shape
    print("Test on target " + target)
    model_manager = ModelManager(model, preprocess)
    y_test, y_pred = model_manager.test_model(X,y)
    
    if (preprocess == Y_all_patches_per_latent or preprocess == YUV_all_patches_per_latent):
        print("Majority voting on " + str(H*W) + " patches")
        num_patches_per_img = H*W
        if len(y_pred) % num_patches_per_img != 0:
            print(f"Warning: {len(y_pred)} predictions not divisible by {num_patches_per_img} patches per image")
        num_img = len(y)
        predictions = []
        real_labels = []
        for i in range(num_img):
            
            start = i * num_patches_per_img
            end = start + num_patches_per_img
            X_img = X[start:end]
            y_img = y[start:end]
            y_test, y_pred = model_manager.test_model(X_img, y_img)
            from collections import Counter
            major_vote = Counter(y_pred).most_common(1)[0][0]
            predictions.append(major_vote)
            real_labels.append(y_test[start])
        y_test = np.array(real_labels)
        y_pred = np.array(predictions)
    print(f"len y_test: {len(y_test)}, len y_pred: {len(y_pred)}")

    print(classification_report(y_test, y_pred))

    print("Accuray score : " + str(accuracy_score(y_test, y_pred)))
    print("Confusion matrix : \n" + str(confusion_matrix(y_test, y_pred)))


def test_process(args, preprocess):
    print(f"\n {10*'*'} Testing {10*'*'}")
    num_train_samples = args.num_samples
    args.num_samples = args.num_samples_test if args.num_samples_test is not None else args.num_samples
    save_latent_path = os.path.join(args.bin_path,str(args.set_target_bpp)+"_bpp","test","latent")

    X_raw, X_hat_raw, labels = prepare_dataset(args, args.test_csv, save_latent_path)

    test_single_target(X_raw, labels, args.model_file_y, preprocess, target='y')
    del X_raw
    test_single_target(X_hat_raw, labels, args.model_file_y_hat,preprocess, target='y_hat')

    args.num_samples = num_train_samples

def test_single_target(X,y, model_path,preprocess, target):
    model_file_path = model_path
    model = load_on_RAM(model_file_path)
    print("Loaded model from " + model_file_path)
    #print(model)
    if target=='y':
        testing(model, X, y, preprocess, target=target)
    elif target=='y_hat':
        testing(model, X, y, preprocess, target=target)


if __name__ == "__main__":
    args = setup_parser()  

    args.num_samples_test = 2000

#    args.test_csv = ""
    args.model_file_y = "/data/lesc/users/rustichini/thesis/models_saved/SINGLE_PATCH/YUV/NEW/600_bpp/20000_samples/y/RF_random_search.joblib"
    args.model_file_y_hat = "/data/lesc/users/rustichini/thesis/models_saved/SINGLE_PATCH/YUV/NEW/600_bpp/20000_samples/y_hat/RF_random_search.joblib"

    #test_process(args, preprocess=single_patch_per_latent)
    model = load_on_RAM(args.model_file_y)
    #print("Loaded model from " + args.model_file_y)
    #print(model)
    print(model.best_estimator_)
    print(model.best_score_)
    
