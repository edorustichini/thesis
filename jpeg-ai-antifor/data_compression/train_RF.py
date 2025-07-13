import os, glob
import torch
import pandas as pd
import numpy as np
import argparse

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

from tqdm import tqdm
import pickle
import joblib
from jpegai_compress_directory import * #setup_coder

from PIL import Image

def get_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def get_latents(img_path, coder, target='y'):
    decisions = coder.get_latents(img_path, bin_path=None, dec_save_path=None)

    latent = decisions['CCS_SGMM']

    features = {
            'model_y': latent['model_y'][target],
            'model_uv' :latent['model_uv'][target]
            }
    
    return features



def process_image(path):
    img_path = path
    ext = os.path.splitext(img_path)[1].lower()

    if ext != ".png":
        # Crea path temporaneo in .png
        img_path_png = os.path.splitext(img_path)[0] + ".png"

        # Converti in PNG
        with Image.open(img_path) as im:
            im.convert("RGB").save(img_path_png)

        img_path = img_path_png  # usa il path PNG

        # temporary_files.append(img_path_png)

    return img_path


def create_dataset(coder, dataset,img_dir, target='y'):

    features_set = []
    labels = []
    
    temp_file = []

    for idx, row in tqdm(dataset.iterrows(), total=len(dataset)):
    
        img_path = os.path.join(img_dir, str(row['path']))
        ext = os.path.splitext(img_path)[1].lower()

        if ext != ".png":
            # Crea path temporaneo in .png
            img_path_png = os.path.splitext(img_path)[0] + ".png"

            # Converti in PNG
            with Image.open(img_path) as im:
                im.convert("RGB").save(img_path_png)
            
            img_path = img_path_png  # usa il path PNG
            temp_file.append(img_path)
        
        features = get_latents(img_path, coder)
        features = torch.cat([torch.flatten(features['model_y']), torch.flatten(features['model_uv'])]).cpu().numpy() #TODO controllare .cpu, .numpy()
        features_set.append(features)
        
        label = row['label']
        labels.append(label)
    
    for path in temp_file:
        if os.path.exists(path):
            os.remove(path)


    features_array = np.array(features_set)
    labels_array = np.array(labels)
    return shuffle(features_array, labels_array, random_state=42) #TODO: decidere se 

def save_model(model_obj, save_dir, model_name):
    os.makedirs(save_dir, exist_ok=True) 
    model_path = os.path.join(save_dir, model_name + ".joblib")
    joblib.dump(model_obj, model_path)

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def sample(df, N):
    df_1 = df[df['label']==1].sample(N//2)
    df_0 = df[df['label']==0].sample(N//2)
    df = pd.concat([df_1, df_0])
    return df

def setup(args):
    # --- Setup the device --- #
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Using GPU {args.gpu}")
        args.target_device = 'gpu'

    # --- Setup the coder --- #
    encoder_parser = create_custom_parser(args)

    # --- Setup the encoder --- #
    coder = RecoEncoder(encoder_parser, def_encoder_parser_decorator(encoder_parser))


    # --- Setup the coding engine --- #
    coder.print_coder_info()

    kwargs, params, _ = coder.init_common_codec(build_model=True, ce=None, cmd_args = None,
                                                overload_ce=True, cmd_args_add=False)
    profiler_path = kwargs.get('profiler_path', None)
    # print(params)

    # Load the models
    coder.load_models(get_downloader(kwargs.get('models_dir_name', 'models'), critical_for_file_absence=not kwargs.get('skip_loading_error', False)))
    coder.set_target_bpp_idx(kwargs['bpp_idx'])
    return coder


if __name__=="__main__":
    # --- Setup an argument parser --- #
    # Arguments for coder
    parser = argparse.ArgumentParser(description='Compress a directory of images using the RecoEncoder')
    parser.add_argument('--gpu', type=int, default=None, help='GPU index')
    parser.add_argument('--imgs_path', type=str, default='../../real_vs_fake/real-vs-fake', help='Input directory')
    parser.add_argument('input_path', type=str, default='../../input_imgs', help='Input directory')
    parser.add_argument('bin_path', type=str, default='../../JPEGAI_output/', help='Save directory')
    parser.add_argument('--set_target_bpp', type=int, default=1, help='Set the target bpp '
                                                                      '(multiplied by 100)')
    parser.add_argument('--models_dir_name', type=str, default='../../jpeg-ai-reference-software/models', help='Directory name for the '
                                                                                 'models used in the encoder-decoder'
                                                                                 'pipeline')
    #Arguments for training
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to process')
    parser.add_argument("--train_csv",default="../../train.csv" , help="Path to dataset's csv file")
    parser.add_argument("--test_csv", default="../../test.csv", help="Path to test's csv file")
    parser.add_argument("-t", "--target",default="y", help="y_hat if quantized latent, else y")
    parser.add_argument("--models_save_dir", default="/data/lesc/users/rustichini/thesis/models_saved", help="Directory to save models")
    args = parser.parse_args()

    coder = setup(args)

    imgs_dir = args.imgs_path
    target = args.target #TODO: da provare anche con hat
    
    df_train = pd.read_csv(args.train_csv)
    n_train_samples = args.num_samples
    df_train = sample(df_train,n_train_samples)
    X_train, y_train = create_dataset(coder, df_train,imgs_dir, target)
    
    #print("(num_samples, num_features)")
    #print(f"Train dataset : {X_train.shape}")
    
    save_path = os.path.join(args.models_save_dir,str(n_train_samples) + "_samples",str(target))
    print("Models will be save into "+ save_path)

    print("\nTraining  random forest...")
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    print("Training finished ")
    print("-"*90)
    save_model(rf, save_path, model_name="RF_"+str(rf.n_estimators)+"_estimators")
    
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
    save_model(grid_search, save_path, model_name="grid_search_RF") 
    