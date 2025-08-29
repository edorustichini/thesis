#!/bin/bash
python train_RF.py ../../input_imgs/ ../../JPEGAI_output/ --set_target_bpp 12 --num_samples 50000 --gpu 2 --save True

