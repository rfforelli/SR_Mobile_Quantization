import os
import argparse
import cv2
import numpy as np
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
from solvers.networks.base7 import base7_quantized
import math
from utils import ProgressBar
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
import logging


class Epoch_End_Callback(Callback):
    def __init__(self, val_data, train_data, lg, val_step):
        super(Epoch_End_Callback, self).__init__()
        self.val_step = val_step
        self.val_data = val_data
        self.train_data = train_data
        self.lg = lg
        self.best_psnr = 0
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs):
        
        self.train_data.shuffle()
        if epoch % self.val_step != 0:
            return

        # validate
        psnr = 0.0
        pbar = ProgressBar(len(self.val_data))
        for i, (lr, hr) in enumerate(self.val_data):
            sr = self.model(lr)
            sr_numpy = K.eval(sr)
            psnr += self.calc_psnr((sr_numpy).squeeze(), (hr).squeeze())
            pbar.update('')
        psnr = round(psnr / len(self.val_data), 4)
        loss = round(logs['loss'], 4)

        # save best status
        if psnr > self.best_psnr:
            self.best_psnr = psnr
            self.best_epoch = epoch

        self.lg.info('Epoch: {:4} | PSNR: {:.2f} | Loss: {:.4f} | lr: {:.2e} | Best_PSNR: {:.2f} in Epoch [{}]'.format(epoch, psnr, loss, K.get_value(self.model.optimizer.lr), self.best_psnr, self.best_epoch))

    def calc_psnr(self, y, y_target):
        h, w, c = y.shape
        y = np.clip(np.round(y), 0, 255).astype(np.float32)
        y_target = np.clip(np.round(y_target), 0, 255).astype(np.float32)

        # crop 1
        y_cropped = y[1:h-1, 1:w-1, :]
        y_target_cropped = y_target[1:h-1, 1:w-1, :]
        
        mse = np.mean((y_cropped - y_target_cropped) ** 2)
        if mse == 0:
            return 100
        return 20. * math.log10(255. / math.sqrt(mse))

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

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

    # Tensorboard save directory
    resume = args['solver']['resume']
    tensorboard_path = 'Tensorboard/{}'.format(args['name'])

    if resume==False:
        if osp.exists(tensorboard_path):
            shutil.rmtree(tensorboard_path, True)
            lg.info('Remove dir: [{}]'.format(tensorboard_path))
    writer = SummaryWriter(tensorboard_path)

    # create dataset
    train_data = DIV2K(args['datasets']['train'])
    lg.info('Create train dataset successfully!')
    lg.info('Training: [{}] iterations for each epoch'.format(len(train_data)))
        
    val_data = DIV2K(args['datasets']['val'])
    lg.info('Create val dataset successfully!')
    lg.info('Validating: [{}] iterations for each epoch'.format(len(val_data)))
        
    height = 216
    #args['datasets']['dimensions']['LR']['height']
    width = 680
    #args['datasets']['dimensions']['LR']['width']
    dimensions = [height, width]
    # create solver
    model = base7_quantized()
    model.summary()
    lr = 1e-3
    optimizer = tf.keras.optimizers.Adam(lr=lr)

    # train
    model.compile(optimizer=optimizer, loss='mae')
    epoch_end_call = Epoch_End_Callback(val_data, train_data, lg, val_step=1)
    def lr_scheduler(epoch, lr):        
        if epoch in args['solver']['lr_steps']:
            lr = lr*args['solver']['lr_gamma']
        return lr
    lr_schedule = LearningRateScheduler(lr_scheduler)
    model.fit(train_data, epochs=20, batch_size = 128, callbacks=[epoch_end_call])
