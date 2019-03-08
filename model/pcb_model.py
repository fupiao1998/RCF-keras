from keras.layers import Conv2D, Conv2DTranspose, Input, MaxPooling2D
from keras.layers import Concatenate, Activation, Add
from keras.models import Model
from model.ResNeXt import residual_block
from loss_functions import cross_entropy_balanced, pixel_error
from keras.optimizers import Adam, SGD


def deconv_layer(x, factor):
    x = Conv2D(1, (1, 1), activation=None, padding='same')(x)
    kernel_size = (2*factor, 2*factor)
    x = Conv2DTranspose(1, kernel_size, strides=factor, padding='same', use_bias=False, activation=None)(x)
    return x


def pcb_model(input_shape=None):
    # Input
    inputs = Input(shape=input_shape)  # 256 256 3

    # Block 1
    x1_conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    x1_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x1_conv1)


    # b1 = deconv_layer(x1_conv2, 1) # 480 480 1
    res_out_1 = residual_block(x1_conv2, 12, stride=(1, 1))
    res_out_1 = Conv2D(1, (1, 1), padding='same', use_bias=False, activation=None)(res_out_1)

    x1 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x1_conv2) # 128 128


    # Block 2
    x2_conv1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x1) # 128 128
    x2_conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x2_conv1)


    # b2 = deconv_layer(x2_conv2, 2) # 480 480 1
    res_out_2 = residual_block(x2_conv2, 12, stride=(1, 1))
    res_out_2 = Conv2D(1, (1, 1), padding='same', use_bias=False, activation=None)(res_out_2)

    x2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x2_conv2) # 64 64 128


    # Block 3
    x3_conv1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x2)
    x3_conv2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x3_conv1)
    x3_conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x3_conv2)

    # b3 = deconv_layer(x3_conv3, 4) # 480 480 1
    res_out_3 = residual_block(x3_conv3, 12, stride=(1, 1))
    res_out_3 = Conv2D(1, (1, 1), padding='same', use_bias=False, activation=None)(res_out_3)

    x3 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x3_conv3) # 32 32 256


    # Block 4
    x4_conv1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x3)
    x4_conv2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x4_conv1)
    x4_conv3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x4_conv2)


    # b4 = deconv_layer(x4_conv3, 8) # 480 480 1
    res_out_4 = residual_block(x4_conv3, 12, stride=(1, 1))
    res_out_4 = Conv2D(1, (1, 1), padding='same', use_bias=False, activation=None)(res_out_4)

    x4 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block4_pool')(x4_conv3) # 16 16 512


    # Block 5
    x5_conv1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x4)
    x5_conv2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x5_conv1)
    x5_conv3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x5_conv2) # 16 16 512

    # b5 = deconv_layer(x5_conv3, 16) # 480 480 1
    res_out_5 = residual_block(x5_conv3, 12, stride=(1, 1))
    res_out_5 = Conv2D(1, (1, 1), padding='same', use_bias=False, activation=None)(res_out_5)

    x5 = Conv2D(1, (1,1), padding='same', use_bias=False, activation=None)(x5_conv3) # 16 16

    # Block 6

    x6_conv1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv1')(x5)
    x6_conv2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv2')(x6_conv1)
    x6_conv3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv3')(x6_conv2)  # 16 16 512

    x6_out = Conv2D(1, (1, 1), padding='same', use_bias=False, activation=None)(x6_conv3)
    x6_add = Add()([x6_out, res_out_5])

    x6 = Activation('relu')(x6_add)
    # kernel_size = (2 * factor, 2 * factor)
    x6 = Conv2DTranspose(1, (4, 4), strides=2, padding='same', use_bias=False, activation=None)(x6)  # 32 32 512

    # Block 7
    x7_conv1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block7_conv1')(x6)
    x7_conv2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block7_conv2')(x7_conv1)
    x7_conv3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block7_conv3')(x7_conv2)

    x7_out = Conv2D(1, (1, 1), padding='same', use_bias=False, activation=None)(x7_conv3)
    x7_add = Add()([x7_out, res_out_4])

    x7 = Activation('relu')(x7_add)
    # kernel_size = (2 * factor, 2 * factor)
    x7 = Conv2DTranspose(1, (4, 4), strides=2, padding='same', use_bias=False, activation=None)(x7)  # 64 64 512

    # Block 8
    x8_conv1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block8_conv1')(x7)
    x8_conv2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block8_conv2')(x8_conv1)
    x8_conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block8_conv3')(x8_conv2)

    x8_out = Conv2D(1, (1, 1), padding='same', use_bias=False, activation=None)(x8_conv3)
    x8_add = Add()([x8_out, res_out_3])

    x8 = Activation('relu')(x8_add)
    # kernel_size = (2 * factor, 2 * factor)
    x8 = Conv2DTranspose(1, (4, 4), strides=2, padding='same', use_bias=False, activation=None)(x8)  # 128 128 512

    # Block 9
    x9_conv1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block9_conv1')(x8)
    x9_conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block9_conv2')(x9_conv1)

    x9_out = Conv2D(1, (1, 1), padding='same', use_bias=False, activation=None)(x9_conv2)
    x9_add = Add()([x9_out, res_out_2])

    x9 = Activation('relu')(x9_add)
    # kernel_size = (2 * factor, 2 * factor)
    x9 = Conv2DTranspose(1, (4, 4), strides=2, padding='same', use_bias=False, activation=None)(x9)  # 256 2526 512

    # Block 10
    x10_conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block10_conv1')(x9)
    x10_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block10_conv2')(x10_conv1)  # 16 16 512

    x10_out = Conv2D(1, (1, 1), padding='same', use_bias=False, activation=None)(x10_conv2)
    x10_add = Add()([x10_out, res_out_1])

    x10 = Activation('relu')(x10_add)
    # kernel_size = (2 * factor, 2 * factor)
    out = Conv2D(1, (1, 1), padding='same', use_bias=False, activation=None)(x10)
    out = Activation('sigmoid', name='out')(out)


    # model
    model = Model(inputs=[inputs], outputs=[out])

    return model