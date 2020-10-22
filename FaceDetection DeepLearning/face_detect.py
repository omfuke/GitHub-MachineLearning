from keras.layers import Input, Dense, Conv2D, MaxPooling2D, PReLU, Flatten, Softmax
from keras.models import Model
import numpy as np


def pnet(input_shape=None):
    if input_shape == None:
        input_shape = (None, None, 3)

    p_inp = Input(input_shape)

    p_layer = Conv2D(10, kernel_size=(
        3, 3), strides=(1, 1), padding="valid")(p_inp)
    p_layer = PReLU(shared_axes=[1, 2])(p_layer)
    p_layer = MaxPooling2D(pool_size=(2, 2), strides=(
        2, 2), padding="same")(p_layer)

    p_layer = Conv2D(16, kernel_size=(3, 3), strides=(
        1, 1), padding="valid")(p_layer)
    p_layer = PReLU(shared_axes=[1, 2])(p_layer)

    p_layer = Conv2D(32, kernel_size=(3, 3), strides=(
        1, 1), padding="valid")(p_layer)
    p_layer = PReLU(shared_axes=[1, 2])(p_layer)

    p_layer_out1 = Conv2D(2, kernel_size=(1, 1), strides=(1, 1))(p_layer)
    p_layer_out1 = Softmax(axis=3)(p_layer_out1)

    p_layer_out2 = Conv2D(4, kernel_size=(1, 1), strides=(1, 1))(p_layer)

    p_net = Model(p_inp, [p_layer_out2, p_layer_out1])

    return p_net


pnet = pnet()

ex = np.arange(432).reshape(12, 12, 3)
ex = np.expand_dims(ex, 0)
out = pnet.predict(ex)


out1 = np.transpose(out[0], (0, 2, 1, 3))
out2 = np.transpose(out[1], (0, 2, 1, 3))
print(out1.shape)
print(out2.shape)

print(out2[0, :, :, 1])
print(out1[0, :, :, :])
