
import os
import pandas as pd

from coder import CoderManager
from dataset import DatasetManager


def prepare_dataset(args, df, save_latent_path):
    # Coder setup
    coder_manager = CoderManager(args)
    
    # Dataset setup
    dataset_manager = DatasetManager(coder_manager.coder)
    
    # -- Dataset creation --
    df = pd.read_csv(args.train_csv)

    #Clean dataframe
    df = df.set_index('id')
    df = df.drop(columns=['Unnamed: 0', 'original_path'])
    if args.num_samples is not None:
        df = dataset_manager.sample_subset(df, args.num_samples, args.random_sample)
    
    
    print("Training set's dataframe")
    print(df)

    # -- Extract latents from images --
    X_raw, X_hat_raw, labels = dataset_manager.build_latent_dataset(
        df,
        args.imgs_path,
        save_latent_path)
    
    return X_raw, X_hat_raw, labels

