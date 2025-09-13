import os.path
import pandas as pd
from tqdm import tqdm
import torch
import sys
sys.path.append('../')
from common import save, load_on_RAM
from .preprocessing import create_patches_dataset

class DatasetManager:
    "Class for database management"
    def __init__(self, coder):
        self.coder = coder
        # TODO: valurare se mettere dataframe tra attributi, o path per dir dove salvare cose
        
    def build_latent_dataset(self, df: pd.DataFrame, img_dir:str, save_dir: str= None):
        """
        Returns X, and X_hat, sets containing dicts for latents of the images selected, and their labels
        If save_dir is None, doesn't save the latent on disc
        Args:
            df : dataframe for the dataset, containing images_paths
            img_dir : the path to the images's directory
            save_dir: path to directory for saving latents
        """
        X, X_hat, labels = [],[],[]
        

        if save_dir is not None:
            # Dir to save latents
            latent_y_path = os.path.join(save_dir, 'y')
            os.makedirs(latent_y_path, exist_ok=True)
            latent_y_hat_path = os.path.join(save_dir, 'y_hat')
            os.makedirs(latent_y_hat_path, exist_ok=True)
            print("Saving 'y' in "+latent_y_path)
            print("Saving 'y_hat' in  "+latent_y_hat_path) 
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting latents from selected images..."):
            img_path = os.path.join(img_dir, str(row['path']))
            
            # checks if altready processed
            if save_dir is not None:
                y_file = latent_y_path +"/"+ str(idx)
                y_hat_file = latent_y_hat_path + "/"+str(idx)
                                
                if os.path.exists(y_file + '.joblib') or os.path.exists(y_hat_file+ '.joblib'):
                    data = load_on_RAM(y_file + '.joblib')
                    X.append(data)
                    data_hat = load_on_RAM(y_hat_file + '.joblib')
                    X_hat.append(data_hat)
                    label = row['label']
                    labels.append(label)
                    continue

            decisions = self.coder.get_latents(img_path, bin_path=None, dec_save_path=None)
            
            if decisions is None:
                print(f"Warning: No decisions returned for {row['path']}")
                continue # TODO: gestire
                
            if 'CCS_SGMM' not in decisions:
                print(f"Warning: CCS_SGMM not found in decisions for {row['path']}")
                continue
                    
            latent = decisions['CCS_SGMM'] # for now only CCS_SGMM implemented

            y,y_hat = self.get_both_targets(latent)

            y_cpu = {
                'model_y': y['model_y'].cpu(),
                'model_uv': y['model_uv'].cpu()
            }
            y_hat_cpu = {
                'model_y': y_hat['model_y'].cpu(), 
                'model_uv': y_hat['model_uv'].cpu()
            }
            if save_dir is not None:
                # Saving the "raw" latents
                self.save_latent(latent_y_path, idx, y_cpu)
                self.save_latent(latent_y_hat_path, idx, y_hat_cpu)

            X.append(y_cpu)
            X_hat.append(y_hat_cpu)
            labels.append(row['label'])
        return X, X_hat, labels
    
    def extract_target(self, latent, target: str = 'y') -> dict:
        """
        Returns either target dictionary containing all the latent tensors
        Args:
            latent: decision object create by the encoder
            target: target feature
        """
        latent_target = {
            'model_y': latent['model_y'][target],
            'model_uv': latent['model_uv'][target]
        }
        return latent_target
    
    def get_both_targets(self, latent):
        """
        Returns latent rapresentation y and quantized version y_hat
        Args:
            latent: Decision object create by the encoder
        """
        y = self.extract_target(latent, 'y')
        y_hat = self.extract_target(latent, 'y_hat')
        return y, y_hat
    
    def save_latent(self, path, id, latent):
        save(latent, path, id)
    
    def sample_subset(self, df, N, random: bool = False):
        """
        Creates a dataset's balanced subset of N samples
        """
        if random:
            df_1 = df[df['label'] == 1].sample(N // 2)
            df_0 = df[df['label'] == 0].sample(N // 2)
        else:
            df_1 = df[df['label'] == 1][:N // 2]
            df_0 = df[df['label'] == 0][:N // 2]
        final_df = pd.concat([df_1, df_0])
        return final_df

def prepare_dataset(args, df_path : str, save_latent_path=None):
    # Coder setup
    from coder import CoderManager
    coder_manager = CoderManager(args)
    
    dataset_manager = DatasetManager(coder_manager.coder)

    df = pd.read_csv(df_path)

    #Clean dataframe
    df = df.set_index('id')
    df = df.drop(columns=['Unnamed: 0', 'original_path'])
    if args.num_samples is not None:
        df = dataset_manager.sample_subset(df, args.num_samples, args.random_sample)
    
    print("Dataframe")
    print(df)

    # -- Extract latents from images --
    X_raw, X_hat_raw, labels = dataset_manager.build_latent_dataset(
        df,
        args.imgs_path,
        save_latent_path)
    
    return X_raw, X_hat_raw, labels

if __name__=="__main__":
    from parser import setup_parser
    args = setup_parser()

    X_raw, X_hat_raw, labels = prepare_dataset(args, args.train_csv)
    X, labels = create_patches_dataset(X_raw, labels)


    
