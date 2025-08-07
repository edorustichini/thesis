import os, glob
import torch
import pandas as pd
import numpy as np
import argparse

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA

from tqdm import tqdm
import pickle
import joblib
from jpegai_compress_directory import * #setup_coder
import matplotlib.pyplot as plt


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

def create_dataset(coder: RecoEncoder, dataset: pd.DataFrame,img_dir: str, save_path = None):
    '''
    Create two different (X, y) for both targets
    '''
    print("BEGIN")
    y_set = []
    y_hat_set = []
    labels = []
    
    temp_file = []


    dataset = dataset.set_index('id')
    dataset = dataset.drop(columns=['Unnamed: 0', 'original_path'])
    print(dataset)
    progress_bar = tqdm(dataset.iterrows(), total=len(dataset), desc="Creating dataset", ncols=100)

    latent_y_path = os.path.join(save_path, 'y')
    os.makedirs(latent_y_path, exist_ok=True)
    latent_y_hat_path = os.path.join(save_path, 'y_hat')
    os.makedirs(latent_y_hat_path, exist_ok=True)


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
        
        y_file = latent_y_path +"/"+ str(idx)
        y_hat_file = latent_y_hat_path + "/"+str(idx)
        
        if os.path.exists(y_file + '.npy') and os.path.exists(y_hat_file+ '.npy'):
            data = np.load(y_file + '.npy')
            y_set.append(data)
            data_hat = np.load(y_hat_file + '.npy')
            y_hat_set.append(data_hat)
            label = row['label']
            labels.append(label)
            continue

        latent = get_latents(img_path, coder)
        
        y = get_target(latent, 'y')
        y_flat= torch.cat([torch.flatten(y['model_y']), torch.flatten(y['model_uv'])]).cpu().numpy() #first move it to cpu, then converts in .numpy()
        y_set.append(y_flat)

        y_hat = get_target(latent, 'y_hat')
        y_hat_flat = torch.cat([torch.flatten(y_hat['model_y']), torch.flatten(y_hat['model_uv'])]).cpu().numpy()
        y_hat_set.append(y_hat_flat)
           
        if save_path is not None:
            np.save(y_file, y_flat)
            np.save(y_hat_file, y_hat_flat)

        label = row['label']
        labels.append(label)

    for path in temp_file: #TODO: verificare se necessario
        if os.path.exists(path):
            os.remove(path)

    y_set = np.array(y_set)
    print(y_set.shape)
    y_hat_set = np.array(y_hat_set)
    labels_array = np.array(labels)

    return y_set, y_hat_set, labels_array 


def test_model(args, model, target: str = 'y'):
    coder = setup(args)
    df_test = pd.read_csv(args.test_csv)
    
    if df_test.shape[0] > args.num_samples_test:
        df_test = sample(df_test,args.num_samples_test)

    save_latent_path = os.path.join(args.bin_path,str(args.set_target_bpp)+"_bpp","test","latent")
    print(save_latent_path)
    if target=='y':
        X_test, _, y_test = create_dataset(coder, df_test, args.imgs_path, save_latent_path)
    elif target=='y_hat':
        _, X_test, y_test = create_dataset(coder, df_test, args.imgs_path, save_latent_path)


    print("Test on target " + target)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    print("Accuray score : " + str(accuracy_score(y_test, y_pred)))

    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix (matplotlib only)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = np.unique(y_test)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted label",
        ylabel="True label",
        title=f"Confusion Matrix ({target})"
    )

    # Annotazioni nei quadranti
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    # Salvataggio immagine
    cm_save_path = os.path.join(args.bin_path, str(args.set_target_bpp) + "_bpp", "test", f"confusion_matrix_{target}.png")
    os.makedirs(os.path.dirname(cm_save_path), exist_ok=True)
    plt.savefig(cm_save_path)
    plt.close()
    print(f"Confusion matrix saved to: {cm_save_path}")

def prepare_dataset(args):
    coder = setup(args)
    imgs_dir = args.imgs_path

    df_train = pd.read_csv(args.train_csv)
    if args.num_samples is not None:
        df_train = sample(df_train, args.num_samples)
    
    num_train = df_train.shape[0] 

    save_latent_path = os.path.join(args.bin_path,str(args.set_target_bpp)+"_bpp","latent")
    
    X_train, X_hat_train, y_train = create_dataset(coder, df_train,imgs_dir, save_latent_path)
    
    print("(num_samples, num_features)")
    print(f"Train dataset : {X_train.shape}")

    save_model_path = os.path.join(args.models_save_dir,str(args.set_target_bpp)+"_bpp",str(num_train) + "_samples")
    return X_train, X_hat_train, y_train, save_model_path # TODO: leva tuple e cambia il modo in cui sono restituiti (cambiando anche quindi train models)


def train_models(X_train, y_train, save_path, target: str):
    ''''
    Trains RF on dataset and save in 'save_path' using parameters found with gridsearch on a dataset's subset
    '''
    '''
    save_path = os.path.join(save_path, target)
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    
    print("\nCreating subset for GridSearchCV...")
    X_sub, _, y_sub, _ = train_test_split(
        X_train, y_train, 
        train_size=10000 if len(X_train) > 10000 else 0.1, # 10% of the dataset 
        stratify=y_train,
        random_state=42
    )
    
    print(f"Subset size: {len(X_sub)} samples (original: {len(X_train)})")
    
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_features': [0.05, 0.1, 'sqrt'],
        'max_depth': [20, None],
        'min_samples_split': [10, 20],
        'min_samples_leaf': [2, 5],
        'bootstrap': [True],
    }
    
    print("\nStarting GridSearchCV on subset...")
    rf = RandomForestClassifier(random_state=42, verbose=1)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3, # TODO: 3 o 5??  
        n_jobs=1, 
        verbose=1
    )
    grid_search.fit(X_sub, y_sub)
    
    print("\nBest parameters found:")
    print(grid_search.best_params_)
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    save(grid_search, save_path, name="grid_search_results")

    best_params = grid_search.best_params_
    
    
    if 'n_estimators' in best_params:
        best_params['n_estimators'] = min(200, best_params['n_estimators'] * 2)  # TODO: da controllare
    
    rf_final = RandomForestClassifier(
        **best_params,
        n_jobs=1,  
        random_state=42,
        verbose=1
    )
    
    rf_final.fit(X_train, y_train)
    
    model_name = f"RF_final_{best_params['n_estimators']}trees_{best_params['max_features']}features"
    save(rf_final, save_path, name=model_name)
    
    print(f"\nModels saved to {save_path}")   

    test_model(args, rf_final, target) '''
    save_path = os.path.join(save_path, target)
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    rf = RandomForestClassifier(max_depth=20, max_features=0.1, min_samples_leaf=5,
                       min_samples_split=20, n_estimators=200, n_jobs=1,
                       random_state=42, verbose=1)
    rf.fit(X_train, y_train)
    save(rf, save_path, name="RF_"+str(rf.n_estimators))



def train_process(args):
    X_train, X_hat_train, y_train, save_path= prepare_dataset(args)
    train_models(X_train, y_train, save_path, 'y')
    
    del X_train
    
    train_models(X_hat_train, y_train,  save_path, 'y_hat')



def save(obj, save_dir, name):
    os.makedirs(save_dir, exist_ok=True) 
    model_path = os.path.join(save_dir, name + ".joblib")
    joblib.dump(obj, model_path)

def load(file_path):
    obj = joblib.load(file_path)
    return obj

def sample(df, N, random: bool = False):
    '''
    Creates a subset of N samples 
    '''
    if random:
        df_1 = df[df['label']==1].sample(N//2)
        df_0 = df[df['label']==0].sample(N//2)
    else: 
        df_1= df[df['label']==1][:N//2]
        df_0= df[df['label']==0][:N//2]
    final_df = pd.concat([df_1, df_0])

    return final_df

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
    print("INIZIO")
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
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to train on')
    parser.add_argument('--num_samples_test', type=int, default=300, help='Number of samples to test on')
    parser.add_argument('--random_sample', type=bool, default=False, help='Sample')
    parser.add_argument("--train_csv",default="../../train.csv" , help="Path to dataset's csv file")
    parser.add_argument("--test_csv", default="../../test.csv", help="Path to test's csv file")
    parser.add_argument("-t", "--target",default=None, help="y_hat if quantized latent, else y")
    parser.add_argument("--save", default=False, help="True if wanted to save dataset")
    parser.add_argument("--models_save_dir", default="/data/lesc/users/rustichini/thesis/models_saved", help="Directory to save models")

    args = parser.parse_args()

    #train_process(args)


    model = load("/data/lesc/users/rustichini/thesis/models_saved/12_bpp/70000_samples/y/RF_200.joblib")
    test_model(args, model,'y')

    model = load("/data/lesc/users/rustichini/thesis/models_saved/12_bpp/70000_samples/y_hat/RF_200.joblib")
    test_model(args, model,'y_hat')
