import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, ReLU, Lambda, Add, Activation
from keras import activations
from qkeras import QConv2D
from qkeras.quantizers import quantized_bits, quantized_relu
from tensorflow.keras.models import Model
from qkeras.qlayers import QActivation
from tensorflow.keras.initializers import glorot_normal
import tensorflow.keras.backend as K
import numpy as np


def base7(scale=3, in_height=216, in_width=680, in_channels=3, num_fea=28, m=4, out_channels=3):

    inp = Input(shape=(in_height, in_width, 3))
    upsample_func = Lambda(lambda x_list: tf.concat(x_list, axis=3))
    upsampled_inp = upsample_func([inp]*(scale**2))

    # Feature extraction
    x = Conv2D(num_fea, 3, padding='same', kernel_initializer=glorot_normal(), bias_initializer='zeros')(inp)
    x = Activation(activations.relu)(x)
    for i in range(m):
        x = Conv2D(num_fea, 3, padding='same', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)
        x = Activation(activations.relu)(x)

    # Pixel-Shuffle
    x = Conv2D(out_channels*(scale**2), 3, padding='same', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)
    x = Activation(activations.relu)(x)
    x = Conv2D(out_channels*(scale**2), 3, padding='same', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)
    x = Add()([upsampled_inp, x])

    depth_to_space = Lambda(lambda x: tf.nn.depth_to_space(x, scale))
    out = depth_to_space(x)
    clip_func = Lambda(lambda x: K.clip(x, 0., 255.))
    out = clip_func(out)

    return Model(inputs=inp, outputs=out)

def base7_quantized(scale=3, in_height=216, in_width=680, in_channels=3, num_fea=28, m=4, out_channels=3):
    quantizer_kwargs = {'kernel_quantizer': quantized_bits(5,0,alpha=1),
              'bias_quantizer': quantized_bits(5,0,alpha=1)}
    inp = Input(shape=(in_height, in_width, 3))
    upsample_func = Lambda(lambda x_list: tf.concat(x_list, axis=3))
    upsampled_inp = upsample_func([inp]*(scale**2))

    # Feature extraction
    x = QConv2D(num_fea, 3, padding='same', kernel_initializer=glorot_normal(), bias_initializer='zeros',**quantizer_kwargs)(inp)
    x = QActivation(activation=quantized_relu(5,5))(x)

    for i in range(m):
        x = QConv2D(num_fea, 3, padding='same', kernel_initializer=glorot_normal(), bias_initializer='zeros',**quantizer_kwargs)(x)
        x = QActivation(activation=quantized_relu(5,5))(x)
    # Pixel-Shuffle
    x = QConv2D(out_channels*(scale**2), 3, padding='same', kernel_initializer=glorot_normal(), bias_initializer='zeros',**quantizer_kwargs)(x)
    x = QActivation(activation=quantized_relu(5,5))(x)
    x = QConv2D(out_channels*(scale**2), 3, padding='same', kernel_initializer=glorot_normal(), bias_initializer='zeros',**quantizer_kwargs)(x)
    x = Add()([upsampled_inp, x])

    depth_to_space = Lambda(lambda x: tf.nn.depth_to_space(x, scale))
    out = depth_to_space(x)
    clip_func = Lambda(lambda x: K.clip(x, 0., 255.))
    out = clip_func(out)

    return Model(inputs=inp, outputs=out)

if __name__ == '__main__':
    model = base7()
    print('Params: [{:.2f}]K'.format(model.count_params()/1e3))
