import argparse
import sys
sys.path.append('../')
from utils.params import JPEG_AI_PATH

def setup_parser():
    '''
    Parse the command line arguments
    '''

    # TODO: migliorabile con parametri definiti in qualche file di config

    parser = argparse.ArgumentParser(description='Compress and train')
    
    coder_group = parser.add_argument_group('coder', 'argument group for coder details')
    
    coder_group.add_argument('--gpu', type=int, default=0, help='GPU index')
    coder_group.add_argument('--set_target_bpp', type=int, default=1, help='Set the target bpp '
                                                                           '(multiplied by 100)')
    coder_group.add_argument('--models_dir_name', type=str, default=JPEG_AI_PATH + "/models",
                             help='Directory name for the '
                                  'models used in the encoder-decoder'
                                  'pipeline')
    io_group = parser.add_argument_group('io', 'argument group for input-output directories')
    io_group.add_argument('--input_path', type=str, default='../../real_vs_fake/real-vs-fake', help='Input directory')
    io_group.add_argument('--bin_path', type=str, default='../../JPEGAI_output/', help='Save directory')
    
    training_group = parser.add_argument_group('train', 'arguments for training')
    training_group.add_argument('--num_samples', type=int, default=1000, help='Number of samples to train on')
    training_group.add_argument('--num_samples_test', type=int, default=300, help='Number of samples to test on')
    training_group.add_argument('--random_sample', type=bool, default=False, help='Sample')
    training_group.add_argument("--train_csv", default="../../train.csv", help="Path to dataset's csv file")
    training_group.add_argument("--models_save_dir", default="../../models_saved/",
                        help="Directory to save models")
    training_group.add_argument("--model_name", default="RF", help="Model name")

    parser.add_argument("--test_csv", default="../../test.csv", help="Path to test's csv file")
    parser.add_argument("-t", "--target", default=None, help="y_hat if quantized latent, else y")
    parser.add_argument("--model_file")
    
    
    parser.add_argument("--save_latents", default=True, help="If set, latents will be saved")


    args = parser.parse_args()    
    return args

