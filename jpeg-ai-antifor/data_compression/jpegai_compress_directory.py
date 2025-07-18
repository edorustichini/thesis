"""
Small script to encode and decode a list of samples inside a directory
and save the decoded image as a .png file.
The script is based on the src.reco.coders.encoder module of the official JPEG AI reference software (https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/main/src/reco/coders/encoder.py?ref_type=heads)
All rights are served to the original authors of the JPEG AI reference software, with the following license

# Copyright (c) 2010-2022, ITU/ISO/IEC
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
# be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.


Authors:
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
"""

# --- Libraries --- #
import os
import torch
import pickle
import glob
from PIL import Image
import sys
import argparse
from tqdm import tqdm
from multiprocessing import cpu_count
# --- Setup the JPEG-AI software suite --- #
sys.path.append('../')
from utils.params import *
sys.path.append(JPEG_AI_PATH)  # Add the jpeg-ai-reference-software to the path
from src.codec import get_downloader
from src.codec.common import Image
from src.codec.coders import CodecEncoder
from src.codec.coders import (def_encoder_base_parser, def_encoder_parser_decorator)


# --- Helpers functions and classes --- #
class RecoEncoder(CodecEncoder):
    def __init__(self, base_parser, parser_decorator, name='reco'):
        super(RecoEncoder, self).__init__(name, base_parser, parser_decorator)

    def encode_stream(self, params):
        raw_image = Image.read_file(params['input_path'])

        if self.ce.target_device == 'cpu':
            #torch.set_num_threads(1)
            torch.set_num_threads(cpu_count()//2)

        self.rec_image, decisions = self.ce.compress(raw_image)

        self.create_bs(params['bin_path'])
        self.init_ec_module()

        self.ce.encode(self.ec_module, decisions)

        self.close_bs()

        return decisions

    def encode_and_decode(self, input_path: str, bin_path: str, dec_save_path: str):
        # Open the image
        try:
            # Read the image file
            raw_image = Image.read_file(input_path)
            

            # Set target device
            if self.ce.target_device == 'cpu':
                #torch.set_num_threads(1)
                torch.set_num_threads(cpu_count()//2)

            # Encode and decode the image
            self.rec_image, decisions = self.ce.compress(raw_image)

            #TODO: decide wheter to skip or not
            # Save the bitstream 
            self.create_bs(bin_path)
            self.init_ec_module()

            self.ce.encode(self.ec_module, decisions)

            self.close_bs()

            # Save decoded image
            self.rec_image.write_png(dec_save_path)

            return decisions
        except Exception as e:
            print(f"Error while processing {os.path.basename(input_path)} : {e}")
            return None

    def get_latents(self, input_path: str, bin_path: str, dec_save_path: str):
        # Open the image
        try:
            # Read the image file
            raw_image = Image.read_file(input_path)
            

            # Set target device
            if self.ce.target_device == 'cpu':
                #torch.set_num_threads(1)
                torch.set_num_threads(cpu_count()//2)

            # Encode and decode the image
            _, decisions = self.ce.compress(raw_image)

            #TODO: decide wheter to skip or not
            # Save the bitstream 
            '''
            self.create_bs(bin_path)
            self.init_ec_module()

            self.ce.encode(self.ec_module, decisions)

            self.close_bs()'''

            # Save decoded image
            #self.rec_image.write_png(dec_save_path)

            return decisions
        except Exception as e:
            print(f"Error while processing {os.path.basename(input_path)} : {e}")
            return None

def create_custom_parser(args: argparse.Namespace):
    parser = def_encoder_base_parser('Reconstruction')

    # Manually add the arguments from the first parser to the second parser
    for key, value in vars(args).items():
        parser.add_argument(f'--{key}', default=value, type=type(value))

    return parser


def list_images(directory):
    # Define the image formats to look for
    image_formats = ["*.png", "*.PNG", "*.jpeg", "*.JPEG", "*.tiff", "*.TIFF", "*.jpg", "*.JPG", '*.tif', '*.TIF']

    # List to store the image paths
    image_files = []

    # Iterate over each format and collect the image files
    for format in image_formats:
        #TODO: images must be in .png format 
        image_files.extend(glob.glob(os.path.join(directory, format)))
    
    
    return image_files


def setup_coder():
    # --- Setup an argument parser --- #
    parser = argparse.ArgumentParser(description='Compress a directory of images using the RecoEncoder')
    parser.add_argument('--gpu', type=int, default=None, help='GPU index')
    parser.add_argument('input_path', type=str, default='', help='Input directory')
    parser.add_argument('bin_path', type=str, default='', help='Save directory')
    parser.add_argument('--set_target_bpp', type=int, default=1, help='Set the target bpp '
                                                                      '(multiplied by 100)')
    parser.add_argument('--models_dir_name', type=str, default='../models', help='Directory name for the '
                                                                                 'models used in the encoder-decoder'
                                                                                 'pipeline')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to process')
    args = parser.parse_args()


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
    return coder, args

def process_dir_with_encoder(coder: RecoEncoder, input_dir: str, bin_dir: str):

    kwargs, params, _ = coder.init_common_codec(build_model=True, ce=None, cmd_args = None,
                                                overload_ce=True, cmd_args_add=False)
    profiler_path = kwargs.get('profiler_path', None)
    # print(params)

    # Load the models
    coder.load_models(get_downloader(kwargs.get('models_dir_name', 'models'), critical_for_file_absence=not kwargs.get('skip_loading_error', False)))
    coder.set_target_bpp_idx(kwargs['bpp_idx'])
    
    # --- Process the directory --- #

    # Create the directories if they don't exist
    save_dir = os.path.join(bin_dir, f"target_bpp_{kwargs['set_target_bpp']}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Process all the files in the input directory
    # and save the decoded images in the save directory
    for root, _, files in os.walk(input_dir):
        image_files = list_images(root)
        if not image_files:
            continue
        
        sub_save_dir = os.path.join(save_dir, os.path.relpath(root, input_dir)) # subdirectory in the save directory for the particular folder we are saving
        os.makedirs(sub_save_dir, exist_ok=True)
        
        for image_path in tqdm(image_files):
            file = os.path.basename(image_path)
            extension = file.split(".")[-1]
            
            bin_path = os.path.join(sub_save_dir, "bins", file.replace(f'.{extension}', ""))
            os.makedirs(os.path.dirname(bin_path), exist_ok=True)
            
            
            dec_path = os.path.join(sub_save_dir, "dec_imgs", file.replace(f'.{extension}', ".png"))
            os.makedirs(os.path.dirname(dec_path), exist_ok=True)
            

            print("\nProcessing " + image_path) 
            if os.path.exists(bin_path):
                print('Skipping (already decoded)', bin_path)
                print("Finished ")
                #TODO: decidere cosa fare in questo caso
                continue
              
            decisions = coder.encode_and_decode(image_path, bin_path, dec_path)
            iterations = 10
            while decisions is None and iterations > 0:
                decisions = coder.encode_and_decode(image_path, bin_path, dec_path)
            if iterations == 0:
                continue

            save_latents(decisions, sub_save_dir, file)
            print("-"*40 + "\n")
        print("Data saved into " + sub_save_dir)
            

def save_latents(decisions, save_dir, file):
    latents_path = os.path.join(save_dir, "latents", file.replace(f'.{file.split(".")[-1]}', ".pt"))
    os.makedirs(os.path.dirname(latents_path), exist_ok=True)
    img_data = {  # TODO: could create a class to store all this data, so keys are transformed in attributes
        'y': dict({
            'model_y': decisions['CCS_SGMM']['model_y']['y'],
            'model_uv': decisions['CCS_SGMM']['model_uv']['y']}),
        'y_hat': dict({
            'model_y': decisions['CCS_SGMM']['model_y']['y_hat'],
            'model_uv': decisions['CCS_SGMM']['model_uv']['y_hat']})}
    torch.save(img_data, latents_path)
    
if __name__ == "__main__":
    coder, args = setup_coder()
    process_dir_with_encoder(coder, args.input_path, args.bin_path)

