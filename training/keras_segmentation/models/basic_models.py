from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Input,
    MaxPool2D,
    MaxPooling2D,
    ZeroPadding2D,
)

from .config import IMAGE_ORDERING


def vanilla_encoder(input_height=224,  input_width=224):

    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2

    if IMAGE_ORDERING == 'channels_first':
        img_input = Input(shape=(3, input_height, input_width))
    elif IMAGE_ORDERING == 'channels_last':
        img_input = Input(shape=(input_height, input_width, 3))

    x = img_input
    levels = []

    x = (ZeroPadding2D((pad, pad), data_format=IMAGE_ORDERING))(x)
    x = (Conv2D(filter_size, (kernel, kernel),
                data_format=IMAGE_ORDERING, padding='valid'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((pool_size, pool_size), data_format=IMAGE_ORDERING))(x)
    levels.append(x)

    x = (ZeroPadding2D((pad, pad), data_format=IMAGE_ORDERING))(x)
    x = (Conv2D(128, (kernel, kernel), data_format=IMAGE_ORDERING,
         padding='valid'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((pool_size, pool_size), data_format=IMAGE_ORDERING))(x)
    levels.append(x)

    for _ in range(3):
        x = (ZeroPadding2D((pad, pad), data_format=IMAGE_ORDERING))(x)
        x = (Conv2D(256, (kernel, kernel),
                    data_format=IMAGE_ORDERING, padding='valid'))(x)
        x = (BatchNormalization())(x)
        x = (Activation('relu'))(x)
        x = (MaxPooling2D((pool_size, pool_size),
             data_format=IMAGE_ORDERING))(x)
        levels.append(x)

    return img_input, levels

def vanilla_encoder2(input_height=256,  input_width=256):

    if IMAGE_ORDERING == 'channels_first':
        img_input = Input(shape=(3, input_height, input_width))
    elif IMAGE_ORDERING == 'channels_last':
        img_input = Input(shape=(input_height, input_width, 3))

    x = img_input
    levels = []

    num_filters = [64, 128, 256, 512]

    for num_filter in num_filters:
        x = Conv2D(num_filter, (3,3), padding='same', data_format=IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(num_filter, (3,3), padding='same', data_format=IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        levels.append(x)
        x = MaxPool2D((2,2), data_format=IMAGE_ORDERING)(x)
    levels.append(x)

    return img_input, levels
