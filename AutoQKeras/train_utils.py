from data import DIV2K
import os.path as osp
import shutil

from tensorboardX import SummaryWriter


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