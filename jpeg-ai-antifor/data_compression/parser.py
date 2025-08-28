import argparse
import sys

# --- Setup the JPEG-AI software suite --- #
sys.path.append('../')
from utils.params import *

def setup_parser():
    # --- Setup an argument parser --- #
    # Arguments for coder
    parser = argparse.ArgumentParser(description='Compress and train')
    
    coder_group = parser.add_argument_group('coder', 'argument group for coder setup')
    
    coder_group.add_argument('--gpu', type=int, default=0, help='GPU index')
    coder_group.add_argument('--set_target_bpp', type=int, default=1, help='Set the target bpp '
                                                                           '(multiplied by 100)')
    coder_group.add_argument('--models_dir_name', type=str, default=JPEG_AI_PATH + "/models",
                             help='Directory name for the '
                                  'models used in the encoder-decoder'
                                  'pipeline')
    io_group = parser.add_argument_group('io', 'argument group for input-output directories')
    io_group.add_argument('--imgs_path', type=str, default='../../real_vs_fake/real-vs-fake', help='Input directory')
    io_group.add_argument('input_path', type=str, default='../../input_imgs', help='Input directory')
    io_group.add_argument('bin_path', type=str, default='../../JPEGAI_output/', help='Save directory')
    
    # Arguments for training
    training_group = parser.add_argument_group('train', 'arguments for training')
    training_group.add_argument('--num_samples', type=int, default=1000, help='Number of samples to train on')
    training_group.add_argument('--num_samples_test', type=int, default=300, help='Number of samples to test on')
    training_group.add_argument('--random_sample', type=bool, default=False, help='Sample')
    training_group.add_argument("--train_csv", default="../../train.csv", help="Path to dataset's csv file")
    
    # Argumets for test
    parser.add_argument("--test_csv", default="../../test.csv", help="Path to test's csv file")
    parser.add_argument("-t", "--target", default=None, help="y_hat if quantized latent, else y")
    parser.add_argument("--save", default=False, help="True if wanted to save dataset")
    
    parser.add_argument("--models_save_dir", default="/data/lesc/users/rustichini/thesis/models_saved",
                        help="Directory to save models")
    
    args = parser.parse_args()
    
    return args

