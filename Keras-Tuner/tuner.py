import os
import argparse
import cv2
import numpy as np
import sys
sys.path.append('../')
from options import parse
from solvers import Solver
from data import DIV2K
import matplotlib as mpl
import matplotlib.pyplot as plt
import shutil
import os
import os.path as osp
from tensorboardX import SummaryWriter
import tensorflow as tf
from solvers.networks.base7 import MyHyperModel
import keras_tuner as kt
from train_utils import generate_train_data, scheduler
from epoch_end_callbacks import Epoch_End_Callback
import pdb

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

    # train
    lg.info('Start tuning...')
    
    hypermodel = MyHyperModel()
    tensorboard = tf.keras.callbacks.TensorBoard("results/tb_logs")
    tuner = kt.RandomSearch(hypermodel,
                    objective='val_loss',
                    directory='results',
                    project_name='kt_results_1',
                    max_trials=50,)

    epoch_end_call = Epoch_End_Callback(val_data, train_data, lg, val_step=1)
    
    tuner.search(train_data, epochs=20, validation_data = val_data, callbacks=[tensorboard, epoch_end_call])
    tuner.results_summary(1)

    # further train the best model to 200 epochs total
    qmodel = tuner.get_best_models(1)[0]
    qmodel.save('keras_tuner_best_model_20_epochs.h5')
    optimizer = tf.keras.optimizers.Adam(lr=1e-3)
    qmodel.compile(optimizer=optimizer, loss='mae')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, min_lr=0.00001)

    qmodel.fit(train_data, validation_data = val_data,epochs=180, callbacks=[reduce_lr,epoch_end_call,tensorboard])
    qmodel.save('keras_tuner_best_model_200_epochs.h5')