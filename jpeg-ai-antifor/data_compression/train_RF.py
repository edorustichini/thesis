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

def get_latents(img_path, coder):
    decisions = coder.get_latents(img_path, bin_path=None, dec_save_path=None)

    latent = decisions['CCS_SGMM']

    #TODO: cosa succede se non rende niente?
    
    return latent

def get_target(latent, target='y'):
    features = {
            'model_y': latent['model_y'][target],
            'model_uv' :latent['model_uv'][target]
            }
    return features


def create_single_dataset(coder: RecoEncoder, dataset: pd.DataFrame, img_dir: str, target: str):
    '''
    Creates dataset only for the target given
    '''
    result = create_dataset(coder, dataset, img_dir)
    if target=='y':
        return result[0]
    elif target=='y_hat':
        return result[1]


def create_dataset(coder: RecoEncoder, dataset: pd.DataFrame,img_dir: str):
    '''
    Create two different (X, y) for both targets
    '''
    y_set = []
    y_hat_set = []
    labels = []
    
    temp_file = []

    progress_bar = tqdm(dataset.iterrows(), total=len(dataset), desc="Creating dataset", ncols=100)

    for idx, row in progress_bar:
    
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
        #TODO: aggiungere eccezioni se problemi con immagini o latenti
        
        latent = get_latents(img_path, coder)
        
        y = get_target(latent, 'y')
        y_flat= torch.cat([torch.flatten(y['model_y']), torch.flatten(y['model_uv'])]).cpu().numpy() #TODO controllare .cpu, .numpy()
        y_set.append(y_flat)

        y_hat = get_target(latent, 'y_hat')
        y_hat_flat = torch.cat([torch.flatten(y_hat['model_y']), torch.flatten(y_hat['model_uv'])]).cpu().numpy()
        y_hat_set.append(y_hat_flat)

        label = row['label']
        labels.append(label)

    for path in temp_file: #TODO: verificare se necessario
        if os.path.exists(path):
            os.remove(path)

    y_set = np.array(y_set)
    y_set = np.array(y_set)
    labels_array = np.array(labels)
    return (shuffle(y_set, labels_array, random_state=42)),((shuffle(y_hat_set, labels_array, random_state=42)))  

def save(obj, save_dir, name):
    os.makedirs(save_dir, exist_ok=True) 
    model_path = os.path.join(save_dir, name + ".joblib")
    joblib.dump(obj, model_path)

def load(file_path):
    obj = joblib.load(file_path)
    return obj

def sample(df, N):
    '''
    Creates a subset of N samples 
    '''
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

def test(args, n_samples: int):
    df_test = pd.read_csv(args.test_csv)
    n_test_samples = 10
    df_test = sample(df_test,n_test_samples)
    X_test, y_test = create_dataset(coder, df_test, args.imgs_path, args.target)
    save_test_path = "../../models_saved/test"+str(args.num_samples)+".joblib"
    os.makedir(save_test_path, ok_exists=True)    
    with open(save_test_path, "wb") as file:
        joblib.dump((X_test, y_test), file)
        
    model_path = "../../models_saved/6_bpp/10_samples/y/RF_100_estimators.joblib"
    model = load(model_path)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

def prepare_dataset(args, save_models:bool = False):
    coder = setup(args)
    imgs_dir = args.imgs_path
    target = args.target #TODO: da provare anche con hat
    df_train = pd.read_csv(args.train_csv)
    n_train_samples = args.num_samples
    if n_train_samples < 100000:
        df_train = sample(df_train,n_train_samples)
    (X_train, y_train), (X_hat_train, y_hat_train ) = create_dataset(coder, df_train,imgs_dir)
    
    #print("(num_samples, num_features)")
    #print(f"Train dataset : {X_train.shape}")
    
    save_path = os.path.join(args.models_save_dir,str(args.set_target_bpp)+"_bpp",str(n_train_samples) + "_samples",'y')
    save_hat_path = os.path.join(args.models_save_dir,str(args.set_target_bpp)+"_bpp",str(n_train_samples) + "_samples",'y_hat')

    if save_models:
        print("Datasets will be saved into:")    
        print(save_path)
        save((X_train, y_train), save_path, 'dataset')
        print(save_hat_path)
        save((X_hat_train, y_hat_train), save_hat_path, 'dataset')
    return (X_train, y_train), (X_hat_train, y_hat_train ), save_path, save_hat_path


def train_models(X_train, y_train, save_path):
    print("Models will be save into "+ save_path)
    print("\nTraining  random forest...")
    rf = RandomForestClassifier(n_estimators=150, n_jobs=8, random_state=42)
    rf.fit(X_train, y_train)
    print("Training finished ")
    print("-"*90)
    save(rf, save_path, name="RF_"+str(rf.n_estimators)+"_estimators")
    
    print("\nSearching with GridSearchCV...")
    param_grid = {
    'n_estimators': [50,100,150,200],             
    'random_state': [42]             
    }
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf,n_jobs=4, param_grid=param_grid, cv=5, verbose=1)
    grid_search.fit(X_train,y_train)
    print("Training finished ")
    print("-"*90)
    save(grid_search, save_path, name="grid_search")

def train_process(args):
    #(X_train, y_train),  _, save_path, _ = prepare_dataset(args)
    
    
    save_path = "/data/lesc/users/rustichini/thesis/models_saved/6_bpp/15000_samples/y_hat"
    X_train, y_train = load("/data/lesc/users/rustichini/thesis/models_saved/6_bpp/15000_samples/y_hat/dataset.joblib")
    
    print("(num_samples, num_features)")
    #print(f"Train dataset : {X_train.shape}")
    train_models(X_train, y_train, save_path)


if __name__=="__main__":
    # --- Setup an argument parser --- #
    # Arguments for coder
    parser = argparse.ArgumentParser(description='Compress a directory of images using the RecoEncoder')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
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
    parser.add_argument("-t", "--target",default=None, help="y_hat if quantized latent, else y")
    parser.add_argument("--models_save_dir", default="/data/lesc/users/rustichini/thesis/models_saved", help="Directory to save models")
    args = parser.parse_args()

    train_process(args)

