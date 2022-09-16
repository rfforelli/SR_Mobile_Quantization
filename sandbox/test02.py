import cv2
import sys
import numpy as np
import tensorflow as tf
from PIL import Image as im
from tensorflow.keras.layers import Input, Lambda, Add
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

sys.path.append('../')

import solvers.networks.base7

if __name__ == '__main__':

    SCALE=3
    IMG_COUNT=10

    for i in range(1, IMG_COUNT+1):
        i_str = str(i).zfill(4)
        img = cv2.imread('../data/DIV2K/bin/DIV2K_train_LR_bicubic/X3/' + i_str + 'x3.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0) # This is to add one more dimension, batch = 1

        print('INFO: Save: ./test02_X_test_' + i_str + '.npy')
        np.save('test02_X_test_' + i_str + '.npy', img[0])

        print('INFO: Save: ./test02_X_test_' + i_str + '.jpg')
        cv2.imwrite('./test02_X_test_' + i_str + '.jpg', img[0])

        # Load pre-trained model (.PB)
        MODEL_DIR = '../experiment/base7_D4C28_bs16ps64_lr1e-3/best_status'
        model = tf.keras.models.load_model(MODEL_DIR)
        #model.summary()

        # DEBUG: remove later layers (Lambda)
        #model = Model(model.input, model.layers[-6].output)
        #model.summary()

        # Run model prediction
        y_keras = model.predict(img)

        print('INFO: Save: ./test02_y_test_' + i_str + '.npy')
        np.save('test02_y_test_' + i_str + '.npy', y_keras[0])

        print('INFO: Save: ./test02_y_test_' + i_str + '.jpg')
        cv2.imwrite('test02_y_test_' + i_str +'.jpg', y_keras[0])

