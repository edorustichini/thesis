
import os
import pandas as pd

from coder import CoderManager
from dataset import DatasetManager


def prepare_dataset(args):
    # Coder setup
    coder_manager = CoderManager(args)
    
    # Dataset setup
    dataset_manager = DatasetManager(coder_manager.coder)
    
    # -- Dataset creation --
    df = pd.read_csv(args.train_csv)

    #Clean dataframe
    df_train= df.set_index('id')
    df_train= df_train.drop(columns=['Unnamed: 0', 'original_path'])
    if args.num_samples is not None:
        df_train = dataset_manager.sample_subset(df_train, args.num_samples, args.random_sample)
    
    
    print("Training set's dataframe")
    print(df_train)

    # -- Extract latents from images --
    save_latent_path = os.path.join(args.bin_path,str(args.set_target_bpp)+"_bpp","latent") if args.save else None
    X_raw, X_hat_raw, labels = dataset_manager.build_latent_dataset(
        df_train,
        args.imgs_path,
        save_latent_path)
    
    return X_raw, X_hat_raw, labels

