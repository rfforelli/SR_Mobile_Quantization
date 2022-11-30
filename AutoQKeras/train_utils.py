from data import DIV2K
import os.path as osp
import shutil
import math
import numpy as np
from tensorboardX import SummaryWriter
import tensorflow.keras.backend as K



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

def calc_psnr(y, y_target):
    def psnr_func(y, y_target):
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
    
    psnr = 0
    for i, (sr, hr) in enumerate(zip(y, y_target)):
        psnr += psnr_func((sr).squeeze(), (hr).squeeze())
    psnr = round(psnr / len(y), 4)
    return psnr