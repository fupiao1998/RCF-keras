import math
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, add, Activation
from keras.layers import Lambda, concatenate
from keras.regularizers import l2

CARDINALITY        = 8            # 4 or 8 or 16
BASE_WIDTH         = 64
IN_PLANES          = 64
WEIGHT_DECAY       = 5e-4


def bn_relu(x):
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    return x


def group_conv(x,planes,stride):
    h = planes // CARDINALITY
    groups = []
    for i in range(CARDINALITY):
        group = Lambda(lambda z: z[:,:,:, i * h : i * h + h])(x)
        groups.append(Conv2D(h,kernel_size=(3,3),strides=stride,kernel_initializer="he_normal",
            kernel_regularizer=l2(WEIGHT_DECAY),
            padding='same',use_bias=False)(group))
    x = concatenate(groups)
    return x


def residual_block(x, planes=3, stride=(1, 1)):
    D = int(math.floor(planes * (BASE_WIDTH / 64.0)))
    C = CARDINALITY

    shortcut = x

    y = Conv2D(D * C, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer="he_normal",
               kernel_regularizer=l2(WEIGHT_DECAY), use_bias=False)(shortcut)
    y = bn_relu(y)

    y = group_conv(y, D * C, stride)
    y = bn_relu(y)

    y = Conv2D(planes * 4, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer="he_normal",
               kernel_regularizer=l2(WEIGHT_DECAY), use_bias=False)(y)
    y = bn_relu(y)

    if stride != (1, 1) or IN_PLANES != planes * 4:
        shortcut = Conv2D(planes * 4, kernel_size=(1, 1), strides=stride, padding='same',
                          kernel_initializer="he_normal", kernel_regularizer=l2(WEIGHT_DECAY), use_bias=False)(x)
        shortcut = BatchNormalization(momentum=0.9, epsilon=1e-5)(shortcut)

    y = add([y, shortcut])
    y = Activation('relu')(y)
    return y