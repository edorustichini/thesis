import os.path
import pandas as pd
from tqdm import tqdm

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
        progress_bar = tqdm(df.iterrows(), total=len(df), desc="Extracting latents from selected images...")
        self.coder.print_coder_info()
        for idx, row in progress_bar:
            img_path = os.path.join(img_dir, str(row['path']))

            decisions = self.coder.get_latents(img_path, bin_path=None, dec_save_path=None)
            # TODO: gestire caso dove non viene ritornato decisions per qualche problema del coder
            
            latent = decisions['CCS_SGMM']

            y,y_hat = self.get_both_targets(latent)

            if save_dir is not None:
                # TODO: da implementare ricordandosi che qua non sono elaborati i dati
                # salvare in un file con idx dell'immagine nel nome, per dare la possibilità di
                #  controllare se un file con lo stesso nome c'è già per saltare l'elaborazione
                # questo controllo avrebbe però farlo prima del calcolo
                pass

            X.append(y)
            X_hat.append(y_hat)
            labels.append(row['label'])
        return X, X_hat, labels
    
    def extract_target(self, latent, target: str = 'y'):
        """
        Returns either 'y' or 'y_hat' dictionary containing all the latent tensors
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
        y = self.extract_target(latent, 'y')
        y_hat = self.extract_target(latent, 'y_hat')
        
        return y, y_hat
    
    def save_latent(self, save_dir, latent):
        pass #TODO: da fare, ma da capire se farla qui o come
    
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

    #TODO: volendo si può aggiungere read_csv con eccezione se non trova nulla
    

        