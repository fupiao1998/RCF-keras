from keras.layers import Conv2D, Conv2DTranspose, Input
from keras.layers import Concatenate, Activation, Add, Lambda, MaxPooling2D, BatchNormalization
from keras.models import Model
from model.ResNet import identity_block, conv_block


def deconv_layer(x, factor):
    x = Conv2D(1, (1, 1), activation=None, padding='same')(x)
    kernel_size = (2*factor, 2*factor)
    x = Conv2DTranspose(1, kernel_size, strides=factor, padding='same', use_bias=False, activation=None)(x)
    return x


def resnet_rcf(input_shape=None):
    # Input
    inputs = Input(shape=input_shape)
    x = Lambda(lambda x: x / 255, name='pre-process')(inputs)

    x = Conv2D(32, (5, 5), strides=(1, 1), padding='same', name='conv1')(x)
    x = BatchNormalization(axis=-1, name='bn_conv1')(x)
    x = Activation('relu', name='act1')(x)
    # Block 1
    x1_conv1 = conv_block(x, 3, (8, 8, 32), stage=1, block='a', strides=(1, 1))
    x1_conv1_out = Conv2D(21, (1, 1), activation='relu', padding='same', name='block1_conv_o1')(x1_conv1)
    x1_conv2 = identity_block(x1_conv1, 3, (8, 8, 32), stage=1, block='b')
    x1_conv2_out = Conv2D(21, (1, 1), activation='relu', padding='same', name='block1_conv_o2')(x1_conv2)

    x1_add = Add()([x1_conv1_out, x1_conv2_out])
    b1 = deconv_layer(x1_add, 1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x1_add)


    # Block 2
    x2_conv1 = conv_block(x1, 3, (16, 16, 64), stage=2, block='a', strides=(1, 1))
    x2_conv1_out = Conv2D(21, (1, 1), activation='relu', padding='same', name='block2_conv_o1')(x2_conv1)
    x2_conv2 = identity_block(x2_conv1, 3, (16, 16, 64), stage=2, block='b')
    x2_conv2_out = Conv2D(21, (1, 1), activation='relu', padding='same', name='block2_conv_o2')(x2_conv2)
    x2_add = Add()([x2_conv1_out, x2_conv2_out])
    b2 = deconv_layer(x2_add, 2) # 480 480 1

    x2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x2_add)


    # Block 3
    x3_conv1 = conv_block(x2, 3, (64, 64, 256), stage=3, block='a', strides=(1, 1))
    x3_conv1_out = Conv2D(21, (1, 1), activation='relu', padding='same', name='block3_conv_o1')(x3_conv1)
    x3_conv2 = identity_block(x3_conv1, 3, (64, 64, 256), stage=3, block='b')
    x3_conv2_out = Conv2D(21, (1, 1), activation='relu', padding='same', name='block3_conv_o2')(x3_conv2)
    x3_conv3 = identity_block(x3_conv2, 3, (64, 64, 256), stage=3, block='c')
    x3_conv3_out = Conv2D(21, (1, 1), activation='relu', padding='same', name='block3_conv_o3')(x3_conv3)

    x3_add = Add()([x3_conv1_out, x3_conv2_out, x3_conv3_out])
    b3 = deconv_layer(x3_add, 4)

    x3 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x3_add)


    # Block 4
    x4_conv1 = conv_block(x3, 3, (64, 64, 256), stage=4, block='a', strides=(1, 1))
    x4_conv1_out = Conv2D(21, (1, 1), activation='relu', padding='same', name='block4_conv_o1')(x4_conv1)
    x4_conv2 = identity_block(x4_conv1, 3, (64, 64, 256), stage=4, block='b')
    x4_conv2_out = Conv2D(21, (1, 1), activation='relu', padding='same', name='block4_conv_o2')(x4_conv2)
    x4_conv3 = identity_block(x4_conv2, 3, (64, 64, 256), stage=4, block='c')
    x4_conv3_out = Conv2D(21, (1, 1), activation='relu', padding='same', name='block4_conv_o3')(x4_conv3)

    x4_add = Add()([x4_conv1_out, x4_conv2_out, x4_conv3_out])
    b4 = deconv_layer(x4_add, 8)

    x4 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block4_pool')(x4_add)


    # Block 5
    x5_conv1 = conv_block(x4, 3, (64, 64, 256), stage=5, block='a', strides=(1, 1))
    x5_conv1_out = Conv2D(21, (1, 1), activation='relu', padding='same', name='block5_conv_o1')(x5_conv1)
    x5_conv2 = identity_block(x5_conv1, 3, (64, 64, 256), stage=5, block='b')
    x5_conv2_out = Conv2D(21, (1, 1), activation='relu', padding='same', name='block5_conv_o2')(x5_conv2)
    x5_conv3 = identity_block(x5_conv2, 3, (64, 64, 256), stage=5, block='c') # 30 30 512
    x5_conv3_out = Conv2D(21, (1, 1), activation='relu', padding='same', name='block5_conv_o3')(x5_conv3)

    x5_add = Add()([x5_conv1_out, x5_conv2_out, x5_conv3_out])
    b5 = deconv_layer(x5_add, 16)

    # fuse
    fuse = Concatenate(axis=-1)([b1, b2, b3, b4, b5])
    fuse = Conv2D(1, (1,1), padding='same', use_bias=False, activation=None)(fuse)

    # outputs
    o1    = Activation('sigmoid', name='o1')(b1)
    o2    = Activation('sigmoid', name='o2')(b2)
    o3    = Activation('sigmoid', name='o3')(b3)
    o4    = Activation('sigmoid', name='o4')(b4)
    o5    = Activation('sigmoid', name='o5')(b5)
    ofuse = Activation('sigmoid', name='ofuse')(fuse)


    # model
    model = Model(inputs=[inputs], outputs=[o1, o2, o3, o4, o5, ofuse])

    return model