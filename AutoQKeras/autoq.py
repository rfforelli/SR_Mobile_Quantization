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
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3500)])
  except RuntimeError as e:
    print(e)
from qkeras import autoqkeras
import logging
from epoch_end_callbacks import Epoch_End_Callback
from train_utils import generate_train_data, psnr_metric
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
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--mode', type=str,required=True)

    args = parser.parse_args()
    pretrained = True if args.pretrained == 'True' else False
    mode = args.mode
    args, lg = parse(args)

    train_data, val_data = generate_train_data(args, lg)

    if pretrained:
      model = tf.keras.models.load_model("../experiment/base7_D4C28_bs16ps64_lr1e-3/best_status")
    else:
      model = base7()

    optimizer = tf.keras.optimizers.Adam(lr=1e-3)
    model.compile(optimizer=optimizer, loss='mae')

    with open("quantization_config.yaml", "r") as stream:
        quantization_config = yaml.safe_load(stream)
    with open("limit.yaml", "r") as stream:
        limit = yaml.safe_load(stream)
    with open("goal.yaml", "r") as stream:
        goal = yaml.safe_load(stream)

output_dir = 'autoqkeras_{mode}'.format(mode=mode)
if pretrained:
  output_dir = '{}_pretrained'.format(output_dir)
output_dir = './'#remove 
project_dir = 'autoqkeras_random_pretrained'
run_config = {
  "directory":'./',
  "project_name":project_dir,
  "goal": goal,
  "quantization_config": quantization_config,
  "learning_rate_optimizer": False,
  "transfer_weights": pretrained,
  "mode": mode,#"hyperband",#random
  "seed": 42,
  "limit": limit,
  "tune_filters": "none",
  "tune_filters_exceptions": "none",
  #only want to get conv layers
  "layer_indexes": np.append(np.arange(1, len(model.layers) - 5),14),
  "max_trials": 100,
  "executions_per_trial": 1,
}

print("quantizing layers:", [model.layers[i].name for i in run_config["layer_indexes"]])
print("run config: ", run_config)
autoqk = autoqkeras.AutoQKeras(model, metrics=['mae'], **run_config)
epoch_end_call = Epoch_End_Callback(val_data, train_data, lg, val_step=1)
autoqk.fit(train_data, validation_data=val_data, epochs=20, callbacks=[epoch_end_call, tf.keras.callbacks.TensorBoard("{output_dir}/tb_logs".format(output_dir=output_dir))])
autoqk.results_summary(1)

# further train the best model to 200 epochs total
qmodel = autoqk.get_best_models(1)[0]
qmodel.save('{}/autoq_best_model_20_epochs.h5'.format(project_dir))
optimizer = tf.keras.optimizers.Adam(lr=1e-3)
qmodel.compile(optimizer=optimizer, loss='mae')
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=20, min_lr=0.00001)

qmodel.fit(train_data, validation_data = val_data,epochs=180, callbacks=[reduce_lr,epoch_end_call])
qmodel.save('{}/autoq_best_model_200_epochs.h5'.format(project_dir))