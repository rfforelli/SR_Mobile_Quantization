from data import DIV2K
import os.path as osp
import shutil
import math
import numpy as np
from tensorboardX import SummaryWriter
import tensorflow.keras.backend as K
from tensorflow.image import psnr
import tensorflow as tf


def generate_train_data(args, lg):
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

    return (train_data, val_data)


def psnr_metric(y, y_target):
    def log10(x):
        numerator = tf.math.log(x)
        denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator

    b, h, w, c = y_target.shape
    y = tf.cast(tf.keras.backend.clip(tf.round(y), 0, 255.), tf.float32)
    y_target = tf.cast(tf.keras.backend.clip(tf.round(y_target), 0, 255.), tf.float32)

    # crop 1
    y_cropped = y[1:h-1, 1:w-1, :]
    y_target_cropped = y_target[1:h-1, 1:w-1, :]

    mse = tf.reduce_mean((y_cropped - y_target_cropped) ** 2)
    if mse == 0:
        return 100.

    return 20. * log10(255. / tf.math.sqrt(mse))