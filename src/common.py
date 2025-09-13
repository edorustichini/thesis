import os
import joblib
import pandas as pd

def save(obj, save_dir, name):
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, name + ".joblib")
    joblib.dump(obj, model_path)
    
def load_on_RAM(file_path):
    obj = joblib.load(file_path)
    return obj

def get_file_name(path: str):
    return os.path.splitext(os.path.basename(path))[0]

def clean_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    
    """
    dataset = dataset.set_index('id')
    dataset = dataset.drop(columns=['Unnamed: 0', 'original_path'])
    return dataset
