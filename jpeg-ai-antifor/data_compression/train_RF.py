import os, glob
import torch
import pandas as pd
import numpy as np
import argparse

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

from tqdm import tqdm
import pickle

def get_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def create_dataset(dataset,latents_dir, target):
    
    latents_set = []
    labels = []

    dataset = dataset.set_index("id")

    for dir in glob.glob(latents_dir):
        # Process the images
        for file_path in tqdm(glob.glob(os.path.join(dir, "**","*.pt"), recursive=True)):

            img_id = get_name(file_path)
            #print(f"Adding {img_id} to the set")
            

            label = dataset.loc[img_id, "label"]
            if label is None:
                continue
                #TODO: controllare se vuote ecc

            #TODO: forse fare un try
            data = torch.load(file_path, weights_only=False)
            
            latent = data[target]
            latent_tensor = torch.cat([torch.flatten(latent['model_y']), torch.flatten(latent['model_uv'])]).cpu().numpy()
            latents_set.append(latent_tensor)
            labels.append(label)

            
    latents_array = np.array(latents_set)
    labels_array = np.array(labels)
    latents_shuffled, labels_shuffled = shuffle(latents_array, labels_array, random_state=42) #TODO: decidere random_state
    
    return latents_shuffled, labels_shuffled

def save_model(model_obj, save_dir, model_name):
    model_pkl_file = os.path.join(save_dir, model_name+".pkl")
    with open(model_pkl_file, 'wb') as file:  
        pickle.dump(model_obj, file)   

def load_model(model_pkl_file):
    with open(model_pkl_file, 'rb') as file:  
        model = pickle.load(model_pkl_file)
    return model 


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir", help="Path to train' data's latents")
    parser.add_argument("train_csv", help="Path to dataset's csv file")

    parser.add_argument("test_dir", help="Path to train' data's latents")
    parser.add_argument("test_csv", help="Path to test's csv file")

    parser.add_argument("-t", "--target",default="y", help="y_hat if quantized latent, else y")

    args=parser.parse_args()
    

    train_dir = args.train_dir
    df_train = pd.read_csv(args.train_csv)
    df_train = df_train.drop(columns = ["original_path", "path"])
    
 

    test_dir =  args.test_dir
    df_test = pd.read_csv(args.test_csv)
    df_test = df_test.drop(columns = ["original_path", "path"])

    target = args.target #TODO: da provare anche con hat

    X_train, y_train = create_dataset(df_train, train_dir, target)
    X_test, y_test = create_dataset(df_test, test_dir, target)
    
    print("(num_samples, num_features)")
    print(f"Train dataset : {X_train.shape}")
    print(f"Test dataset: {X_test.shape}")
    

    with open("/data/lesc/users/rustichini/thesis/models_saved/medium_dataset/y/simple_RF_250.pkl", "rb") as file:
        model = pickle.load(file)
    y_pred_rf = model.predict(X_test)

    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print(f"1. Accuracy on test set using simple RF: {accuracy_rf:.4f}")

    
    '''
    print("\nTraining simple random forest...")
    rf = RandomForestClassifier(n_estimators=250, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    print("Training finished ")
    print("-"*90)
    save_model(rf,"/data/lesc/users/rustichini/thesis/models_saved/medium_dataset/y", "simple_RF_250")

    print("\nTraining HGB...")
    hgb = HistGradientBoostingClassifier(random_state=42)
    hgb.fit(X_train, y_train)
    print("Training finished ")
    print("-"*90)
    save_model(hgb, "/data/lesc/users/rustichini/thesis/models_saved/medium_dataset/y", "hgb")
    
    print("\nSearching with GridSearchCV...")
    param_grid = {
    'n_estimators': [50,100, 200],             
    'max_depth': [None, 10, 20, 50],             
    'max_features': ['sqrt'],       
    'min_samples_split': [2, 5, 10],             
    'min_samples_leaf': [1, 2, 5],               
    }
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(rf, param_grid, cv=5)
    grid_search.fit(X_train,y_train)
    print("Training finished ")
    print("-"*90)
    save_model(grid_search, "/data/lesc/users/rustichini/thesis/models_saved/medium_dataset/y", "grid_search_RF")
    '''