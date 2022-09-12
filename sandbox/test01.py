import cv2
import numpy as np
import tensorflow as tf
from PIL import Image as im
from tensorflow.keras.layers import Input, Lambda, Add
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

def upsample_img(img, scale):
    N, H, W, C = img.shape
    X = img.astype(float)
    #print('N', N, ', H', H, ', W', W, ', C', C)
    #print(img.shape)

    inputs = Input(shape=(None, None, C))
    upsample_func = Lambda(lambda x_list: tf.concat(x_list, axis=3))
    outputs = upsample_func([inputs]*(scale**2))

    model = Model(inputs=inputs, outputs=outputs)

    Y = model.predict(X)

    #print(model.summary())

    return Y

def depth_to_space_img(img, scale):
    N, H, W, C = img.shape
    X = img.astype(float)
    #print('N', N, ', H', H, ', W', W, ', C', C)
    #print(img.shape)

    X = img.astype(float)

    inputs = Input(shape=(None, None, C))
    depth_to_space = Lambda(lambda x: tf.nn.depth_to_space(x, scale))
    outputs = depth_to_space(inputs)
    model = Model(inputs=inputs, outputs=outputs)

    Y = model.predict(X)

    return Y

def add_img(img1, img2):
    N1, H1, W1, C1 = img1.shape
    N2, H2, W2, C2 = img2.shape
    #print('N1', N1, ', H1', H1, ', W1', W1, ', C1', C1)
    #print('N2', N2, ', H2', H2, ', W2', W2, ', C2', C2)
    #print(img1.shape)
    #print(img2.shape)

    X1 = img1.astype(float)
    X2 = img2.astype(float)

    inputs1 = Input(shape=(None, None, C1))
    inputs2 = Input(shape=(None, None, C2))
    outputs = Add()([inputs1, inputs2])
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)

    Y = model.predict([X1, X2])

    return Y


def clip_img(img):
    N, H, W, C = img.shape
    X = img.astype(float)
    #print('N', N, ', H', H, ', W', W, ', C', C)
    #print(img.shape)

    X = img.astype(float)

    inputs = Input(shape=(None,None, C))
    clip_func = Lambda(lambda x: K.clip(x, 0., 255.))
    outputs = clip_func(inputs)
    model = Model(inputs=inputs, outputs=outputs)

    Y = model.predict(X)

    return Y

if __name__ == '__main__':

    SCALE=3

    if False:
        N, H, W, C = [1, 2, 2, 1]
        img = np.array([1,2,3,4]).reshape(N, H, W, C)
        fltr = np.array([10,20,30,40]).reshape(N, H, W, C)

        up_img = upsample_img(img, scale=2)
        add_img = add_img(up_img, fltr)
        ds_img = depth_to_space_img(add_img, scale=2)
        c_img = clip_img(ds_img)

        print(img)
        print(fltr)
        print(up_img)
        print(add_img)
        print(ds_img)
        print(c_img)
    else:
        for i in range(1, 2):
            i_str = str(i).zfill(4)
            img = cv2.imread('../data/DIV2K/bin/DIV2K_train_LR_bicubic/X3/' + i_str + 'x3.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            fltr = np.ones((1, img.shape[0], img.shape[1], img.shape[2]*(SCALE**2)))+125

            img = np.expand_dims(img, axis=0) # This is to add one more dimension, batch = 1

            print('INFO: Save: ./X_test.npy')
            np.save('X_test.npy', img)

            up_img = upsample_img(img, scale=SCALE)
            add_img = add_img(up_img, fltr)
            ds_img = depth_to_space_img(add_img, scale=3)
            c_img = clip_img(ds_img)

            print('INFO: Save: ./input_' + i_str + '.jpg')
            cv2.imwrite('./input_' + i_str + '.jpg', img[0])
            print('INFO: Save: ./output_' + i_str + '.jpg')
            cv2.imwrite('./output_' + i_str + '.jpg', c_img[0])

            ##print(img)
            ##print(c_img)

            #print(img.shape)
            #print(up_img.shape)
            #print(ds_img.shape)
            #print(c_img.shape)

