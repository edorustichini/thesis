import os, glob
import torch
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_spli, GridSearchCV




def get_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def create_dataset(dataset,latents_dir, target = 'y'):
    
    latents_set = []
    labels = []

    for dir in glob.glob(latents_dir):
        # Process the images
        for file_path in glob.glob(os.path.join(dir, "**", "latents", "*.pt")):

            img_id = get_name(file_path)
            print(f"Processing {file_path}")
            

            label = dataset.query('id == @img_id')['label'].iloc[0]
            #TODO: controllare se vuote ecc

            #TODO: forse fare un try
            data = torch.load(file_path)

            latent = data[target]
            latent_tensor = torch.cat([torch.flatten(latent['model_y']), torch.flatten(latent['model_uv'])]).cpu().numpy()
            
            latents_set.append(latent_tensor)
            labels.append(label)

            
    latents_array = np.array(latents_set)
    labels_array = np.array(labels)
    latents_shuffled, labels_shuffled = shuffle(latents_array, labels_array, random_state=42) #TODO: decidere random_state
    
    return latents_shuffled, labels_shuffled



if __name__=="__main__":    
    dir_train = "/data/lesc/users/rustichini/thesis/output/small_dataset/target_bpp_6/train/"
    df_train = pd.read_csv("/data/lesc/users/rustichini/thesis/real-vs-fake/small_dataset/train.csv")
    df_train = df_train.drop(columns = ["original_path", "path"])
    
 

    dir_test = "/data/lesc/users/rustichini/thesis/output/small_dataset/target_bpp_6/test"
    df_test = pd.read_csv("/data/lesc/users/rustichini/thesis/real-vs-fake/small_dataset/test.csv")
    df_test = df_test.drop(columns = ["original_path", "path"])
    
    df_train = df_train.sample(n=3000)
    df_test = df_test.sample(n=850)

    X_train, y_train = create_dataset(df_train, dir_train)
    X_test, y_test = create_dataset(df_test, dir_test)

    
    print(len(X_train), len(X_test))

    
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=0)
    rf.fit(X_train,y_train)
    
    
    # --- Predizioni sul Set di Test ---
    print("\nEsecuzione delle predizioni sul set di TEST...")
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)

    # --- Valutazione delle Prestazioni ---
    print("\n--- Valutazione delle Prestazioni del Random Forest ---")

    num_classes = len(np.unique(y_train))
    class_names = [f'Classe {i}' for i in range(num_classes)]

    accuracy = accuracy_score(y_test, y_pred)
    print(f"1. Accuratezza sul Test Set: {accuracy:.4f}")

    
    
    
    

    