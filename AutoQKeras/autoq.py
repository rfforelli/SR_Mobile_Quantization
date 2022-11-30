# setting path
import sys
sys.path.append('../../SR_Mobile_Quantization')

import os
import argparse
import cv2
import numpy as np
from options import parse
from solvers import Solver
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from qkeras import autoqkeras
import logging
from epoch_end_callbacks import Epoch_End_Callback
from train_utils import generate_train_data, calc_psnr
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import yaml
from solvers.networks.base7 import base7_quantized
from solvers.networks.base7 import base7
import logging

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='FSRCNN Demo')
    parser.add_argument('--opt', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--scale', default=3, type=int)
    parser.add_argument('--ps', default=48, type=int, help='patch_size')
    parser.add_argument('--bs', default=16, type=int, help='batch_size')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--gpu_ids', default=None)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--resume_path', default=None)
    parser.add_argument('--qat', action='store_true', default=False)
    parser.add_argument('--qat_path', default=None)

    args = parser.parse_args()
    args, lg = parse(args)

    train_data, val_data = generate_train_data(args, lg)
    
    #model = tf.keras.models.load_model("../experiment/base7_D4C28_bs16ps64_lr1e-3/best_status")

    # model = base7_quantized()
    model = base7()
    optimizer = tf.keras.optimizers.Adam(lr=1e-3)
    model.compile(optimizer=optimizer, loss='mae')

    model.summary()
    with open("quantization_config.yaml", "r") as stream:
        quantization_config = yaml.safe_load(stream)
    with open("limit.yaml", "r") as stream:
        limit = yaml.safe_load(stream)
    with open("goal.yaml", "r") as stream:
        goal = yaml.safe_load(stream)

run_config = {
  "output_dir": 'autoqkeras_bayesian_scratch',
  "goal": goal,
  "quantization_config": quantization_config,
  "learning_rate_optimizer": False,
  "transfer_weights": False,
  "mode": "random",#"baseyian","hyperband",#random
  "seed": 42,
  "limit": limit,
  "tune_filters": "none",
    "tune_filters_exceptions": "none",
  # first layer is input, layer two layers are softmax and flatten
  "layer_indexes": np.append(np.arange(1, len(model.layers) - 5),8),#range(1, len(model.layers) - 5) ,
  "max_trials": 100
}

print("quantizing layers:", [model.layers[i].name for i in run_config["layer_indexes"]])
autoqk = autoqkeras.AutoQKeras(model, **run_config)
epoch_end_call = Epoch_End_Callback(val_data, train_data, lg, val_step=1)
autoqk.fit(train_data, validation_data=val_data, batch_size=16, epochs=20, callbacks=[epoch_end_call])