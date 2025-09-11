import os.path
import pandas as pd
from tqdm import tqdm
import torch
from common import save, load_on_RAM
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
            # TODO: gestire caso dove non viene ritornato decisions per qualche problema del coder
            
            # Check if decisions is None or empty
            if decisions is None:
                print(f"Warning: No decisions returned for {row['path']}")
                #failed_images.append(row['path'])
                continue
                
            if 'CCS_SGMM' not in decisions:
                print(f"Warning: CCS_SGMM not found in decisions for {row['path']}")
                #failed_images.append(row['path'])
                continue
                    
            latent = decisions['CCS_SGMM']

            y,y_hat = self.get_both_targets(latent)

            # Move torch tensors on CPU
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
                save(y_cpu, latent_y_path, idx)
                save(y_hat_cpu, latent_y_hat_path, idx)

            X.append(y_cpu)
            X_hat.append(y_hat_cpu)
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

def flatten_latents(latents, labels):
    flat_set = []
    for latent in latents:
        flat = torch.cat([torch.flatten(latent['model_y']), torch.flatten(latent['model_uv'])]).cpu().numpy()
        flat_set.append(flat)
    return flat_set, labels


#def channels_to_tensor(latents,labels, channels_per_image=5):

    # TODO: Aggiungi descrizione
    if len(latents) != len(labels):
        raise ValueError("Latents and labels must have same size")

    X, new_labels = [],[]

    k = channels_per_image

    for latent,label in zip(latents,labels):
        #print(latent['model_y'].shape)
        latent_y = latent['model_y'][0] # to remove batch size
        latent_uv = latent['model_uv'][0] # TODO: prendere anche UV concatenando?

        latent = latent_y
        #print(latent_y.shape)
        #print(latent_uv.shape)

        #print(latent.shape)
        chls_idxs = torch.randperm(latent.shape[0])[:k]
        #print(chls_idxs)

        for ch in chls_idxs:
            ch_grid = latent[ch]
            flat = torch.flatten(ch_grid).numpy()
            X.append(flat)
            new_labels.append(label)
            #print(label, end=" ")
        #print("\n")
    return X, new_labels

#def channels_to_tensor_for_testing(latents, labels, channels_per_image=14):
    """
    Versione per testing che mantiene la struttura per immagine
    per permettere il voto a maggioranza
    
    Returns:
        X_grouped: lista di liste, dove ogni sottolista contiene i canali di un'immagine
        labels: labels originali (una per immagine)
    """
    if len(latents) != len(labels):
        raise ValueError("Latents and labels must have same size")

    X_grouped = []
    k = channels_per_image

    for latent, label in zip(latents, labels):
        latent_y = latent['model_y'][0]  # rimuovi batch size
        latent_uv = latent['model_uv'][0]  # TODO: considera anche UV se necessario
        
        latent = latent_y
        
        # Estrai k canali casuali
        chls_idxs = torch.randperm(latent.shape[0])[:k]
        
        # Lista per i canali di questa immagine
        image_channels = []
        for ch in chls_idxs:
            ch_grid = latent[ch]
            flat = torch.flatten(ch_grid).numpy()
            image_channels.append(flat)
        
        X_grouped.append(image_channels)
    
    return X_grouped, labels

def get_single_patch(latent, row, col):
    """
    Given a latent tensor of shape (C, H ,W) returns the concatenations of patches in (x,y) position
    of shape (C, 1, 1)
    Args:
        latent: tensor of shape (C, H, W)
    Returns:
        patch: tensor of shape (C, 1, 1) 
    """
    C, H, W = latent.shape
    
    if row >= H or col >= W:
        raise ValueError("row and col must be less than H and W respectively")
    
    patch = latent[:, row, col]  
    return patch

def create_patches_dataset(latents, labels, patch_size=1):
    """
    Creates a dataset of patches from the latents
    Args:
        latents: list of latent tensors of shape (1,C, H, W)
        labels: list of labels
        patch_size: size of the patch (default 1x1)
    Returns:
        X_patches: list of patches of shape (C*patch_size*patch_size,)
        y_patches: list of labels for each patch
    """
    X_patches = []
    y_patches = []

    for latent, label in zip(latents, labels):

        latent_y = latent['model_y'][0] # to remove batch size
        latent_uv = latent['model_uv'][0]
        
        patch_y  = get_single_patch(latent_y, 8,8)  
        patch_uv = get_single_patch(latent_uv, 8,8)


        patch = torch.cat((torch.flatten(patch_y), torch.flatten(patch_uv))).numpy()
        X_patches.append(patch)
        y_patches.append(label)
    
    print(f"Created {len(X_patches)} patches from {len(latents)} latents.")
    print(f"Each patch has shape: {X_patches[0].shape}")
    print(f"Labels shape: {len(y_patches)}")
    
    return X_patches, y_patches

#def create_test_patches_dataset(latents, labels, patch_size=1):
    """
    Creates a dataset of patches from the latents
    Args:
        latents: list of latent tensors of shape (1,C, H, W)
        labels: list of labels
        patch_size: size of the patch (default 1x1)
    Returns:
        X_grouped: list of lists, where each sublist contains patches for one image
        y_grouped: list of labels for each image
    """
    X_grouped = []
    y_grouped = []

    for latent, label in zip(latents, labels):

        latent_y = latent['model_y'][0] # to remove batch size
        latent_uv = latent['model_uv'][0]
        
        patch_y  = get_single_patch(latent_y, 8,8)  
        patch_uv = get_single_patch(latent_uv, 8,8)


        patch = torch.cat((torch.flatten(patch_y), torch.flatten(patch_uv))).numpy()
        
        X_grouped.append([patch])  # Wrap patch in a list to maintain structure
        y_grouped.append(label)
    
    print(f"Created {len(X_grouped)} grouped patches from {len(latents)} latents.")
    print(f"Each patch has shape: {X_grouped[0][0].shape}")
    print(f"Labels shape: {len(y_grouped)}")
    
    return X_grouped, y_grouped

if __name__=="__main__":
    from parser import setup_parser
    args = setup_parser()

    from main import prepare_dataset
    X_raw, X_hat_raw, labels = prepare_dataset(args, args.train_csv)
    X, labels = create_patches_dataset(X_raw, labels)


    
