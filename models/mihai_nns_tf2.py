import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization, add, concatenate, Lambda, Flatten, Conv2DTranspose, ZeroPadding2D, Cropping2D
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras.backend as K

def map2map_generic(input_shape=(256, 256, 3), init_nb=16, batch_norm=True, activation=None, num_classes=1, name=''):
    
    inputs = Input(input_shape)
    
    conv1 = Conv2D(init_nb, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(inputs)
    if batch_norm:
        conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv2D(init_nb, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(conv1)
    if batch_norm:
        conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv3 = Conv2D(init_nb, (3, 3), padding='same', strides=(2, 2))(conv2)
    if batch_norm:
        conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    
    conv4 = Conv2D(init_nb * 2, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(conv3)
    if batch_norm:
        conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv5 = Conv2D(init_nb * 2, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(conv4)
    if batch_norm:
        conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv6 = Conv2D(init_nb * 2, (3, 3), padding='same', strides=(2, 2))(conv5)
    if batch_norm:
        conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    conv7 = Conv2D(init_nb * 4, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(conv6)
    if batch_norm:
        conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv8 = Conv2D(init_nb * 4, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(conv7)
    if batch_norm:
        conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv9 = Conv2D(init_nb * 4, (3, 3), padding='same', strides=(2, 2))(conv8)
    if batch_norm:
        conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    
    # stacked dilated convolution
    conv10 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=1)(conv9)
    if batch_norm:
        conv10 = BatchNormalization()(conv10)
    conv10 = Activation('relu')(conv10)
    conv11 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=2)(conv10)
    if batch_norm:
        conv11 = BatchNormalization()(conv11)
    conv11 = Activation('relu')(conv11)
    conv12 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=4)(conv11)
    if batch_norm:
        conv12 = BatchNormalization()(conv12)
    conv12 = Activation('relu')(conv12)
    conv13 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=8)(conv12)
    if batch_norm:
        conv13 = BatchNormalization()(conv13)
    conv13 = Activation('relu')(conv13)
    conv14 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=16)(conv13)
    if batch_norm:
        conv14 = BatchNormalization()(conv14)
    conv14 = Activation('relu')(conv14)
    conv15 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=32)(conv14)
    if batch_norm:
        conv15 = BatchNormalization()(conv15)
    conv15 = Activation('relu')(conv15)
    
    dilate_all_added = add([conv10, conv11, conv12, conv13, conv14, conv15])
    
    conv16 = Conv2DTranspose(init_nb * 4, (3, 3), padding='same', strides=(2, 2))(dilate_all_added)
    if batch_norm:
        conv16 = BatchNormalization()(conv16)
    conv16 = Activation('relu')(conv16)
    conv16 = concatenate([conv7, conv16])
    conv17 = Conv2D(init_nb * 4, (3, 3), padding='same')(conv16)
    if batch_norm:
        conv17 = BatchNormalization()(conv17)
    conv17 = Activation('relu')(conv17)
    
    conv18 = Conv2DTranspose(init_nb * 2, (3, 3), padding='same', strides=(2, 2))(conv17)
    if batch_norm:
        conv18 = BatchNormalization()(conv18)
    conv18 = Activation('relu')(conv18)
    conv18 = concatenate([conv4, conv18])
    conv19 = Conv2D(init_nb, (3, 3), padding='same')(conv18)
    if batch_norm:
        conv19 = BatchNormalization()(conv19)
    conv19 = Activation('relu')(conv19)
    
    conv20 = Conv2DTranspose(init_nb, (3, 3), padding='same', strides=(2, 2))(conv19)
    if batch_norm:
        conv20 = BatchNormalization()(conv20)
    conv20 = Activation('relu')(conv20)
    conv20 = concatenate([conv1, conv20])
    conv21 = Conv2D(init_nb, (3, 3), padding='same')(conv20)
    if batch_norm:
        conv21 = BatchNormalization()(conv21)
    conv21 = Activation('relu')(conv21)
    
    conv22 = Conv2D(num_classes, (1, 1), strides=(1, 1), activation=activation)(conv21)

    model = Model(inputs=inputs, outputs=conv22, name=name)

    return model

def map2map_generic_same_initializer(input_shape=(256, 256, 3), init_nb=16, batch_norm=True, activation=None, num_classes=1, name=''):
    
    inputs = Input(input_shape)
    
    conv1 = Conv2D(init_nb, (3, 3), padding='same', strides=(1, 1), dilation_rate=1, kernel_initializer='he_uniform')(inputs)
    if batch_norm:
        conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv2D(init_nb, (3, 3), padding='same', strides=(1, 1), dilation_rate=1, kernel_initializer='he_uniform')(conv1)
    if batch_norm:
        conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv3 = Conv2D(init_nb, (3, 3), padding='same', strides=(2, 2), kernel_initializer='he_uniform')(conv2)
    if batch_norm:
        conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    
    conv4 = Conv2D(init_nb * 2, (3, 3), padding='same', strides=(1, 1), dilation_rate=1, kernel_initializer='he_uniform')(conv3)
    if batch_norm:
        conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv5 = Conv2D(init_nb * 2, (3, 3), padding='same', strides=(1, 1), dilation_rate=1, kernel_initializer='he_uniform')(conv4)
    if batch_norm:
        conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv6 = Conv2D(init_nb * 2, (3, 3), padding='same', strides=(2, 2), kernel_initializer='he_uniform')(conv5)
    if batch_norm:
        conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    conv7 = Conv2D(init_nb * 4, (3, 3), padding='same', strides=(1, 1), dilation_rate=1, kernel_initializer='he_uniform')(conv6)
    if batch_norm:
        conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv8 = Conv2D(init_nb * 4, (3, 3), padding='same', strides=(1, 1), dilation_rate=1, kernel_initializer='he_uniform')(conv7)
    if batch_norm:
        conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv9 = Conv2D(init_nb * 4, (3, 3), padding='same', strides=(2, 2), kernel_initializer='he_uniform')(conv8)
    if batch_norm:
        conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    
    # stacked dilated convolution
    conv10 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=1, kernel_initializer='he_uniform')(conv9)
    if batch_norm:
        conv10 = BatchNormalization()(conv10)
    conv10 = Activation('relu')(conv10)
    conv11 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=2, kernel_initializer='he_uniform')(conv10)
    if batch_norm:
        conv11 = BatchNormalization()(conv11)
    conv11 = Activation('relu')(conv11)
    conv12 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=4, kernel_initializer='he_uniform')(conv11)
    if batch_norm:
        conv12 = BatchNormalization()(conv12)
    conv12 = Activation('relu')(conv12)
    conv13 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=8, kernel_initializer='he_uniform')(conv12)
    if batch_norm:
        conv13 = BatchNormalization()(conv13)
    conv13 = Activation('relu')(conv13)
    conv14 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=16, kernel_initializer='he_uniform')(conv13)
    if batch_norm:
        conv14 = BatchNormalization()(conv14)
    conv14 = Activation('relu')(conv14)
    conv15 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=32, kernel_initializer='he_uniform')(conv14)
    if batch_norm:
        conv15 = BatchNormalization()(conv15)
    conv15 = Activation('relu')(conv15)
    
    dilate_all_added = add([conv10, conv11, conv12, conv13, conv14, conv15])
    
    conv16 = Conv2DTranspose(init_nb * 4, (3, 3), padding='same', strides=(2, 2), kernel_initializer='he_uniform')(dilate_all_added)
    if batch_norm:
        conv16 = BatchNormalization()(conv16)
    conv16 = Activation('relu')(conv16)
    conv16 = concatenate([conv7, conv16])
    conv17 = Conv2D(init_nb * 4, (3, 3), padding='same', kernel_initializer='he_uniform')(conv16)
    if batch_norm:
        conv17 = BatchNormalization()(conv17)
    conv17 = Activation('relu')(conv17)
    
    conv18 = Conv2DTranspose(init_nb * 2, (3, 3), padding='same', strides=(2, 2), kernel_initializer='he_uniform')(conv17)
    if batch_norm:
        conv18 = BatchNormalization()(conv18)
    conv18 = Activation('relu')(conv18)
    conv18 = concatenate([conv4, conv18])
    conv19 = Conv2D(init_nb, (3, 3), padding='same', kernel_initializer='he_uniform')(conv18)
    if batch_norm:
        conv19 = BatchNormalization()(conv19)
    conv19 = Activation('relu')(conv19)
    
    conv20 = Conv2DTranspose(init_nb, (3, 3), padding='same', strides=(2, 2), kernel_initializer='he_uniform')(conv19)
    if batch_norm:
        conv20 = BatchNormalization()(conv20)
    conv20 = Activation('relu')(conv20)
    conv20 = concatenate([conv1, conv20])
    conv21 = Conv2D(init_nb, (3, 3), padding='same', kernel_initializer='he_uniform')(conv20)
    if batch_norm:
        conv21 = BatchNormalization()(conv21)
    conv21 = Activation('relu')(conv21)
    
    conv22 = Conv2D(num_classes, (1, 1), strides=(1, 1), activation=activation, kernel_initializer='he_uniform')(conv21)

    model = Model(inputs=inputs, outputs=conv22, name=name)

    return model

def special_concatenate_crop(layer_1, layer_2, value):
    #output_layer = ZeroPadding2D(padding=((0, 0), (0, value)))(layer_1)
    #return concatenate([output_layer, layer_2])
    output_layer = Cropping2D(cropping=((0, value), (0, 0)))(layer_2)
    return concatenate([layer_1, output_layer])

def special_concatenate_pad(layer_1, layer_2, value):
    output_layer = ZeroPadding2D(padding=((0, value), (0, 0)))(layer_1)
    return concatenate([output_layer, layer_2])
    #output_layer = Cropping2D(cropping=((0, 0), (0, 1)))(layer_2)
    #return concatenate([layer_1, output_layer])

def map2map_generic_same_initializer_new_size(input_shape=(256, 256, 3), init_nb=16, batch_norm=True, activation=None, num_classes=1, name=''):
    
    inputs = Input(input_shape)
    
    conv1 = Conv2D(init_nb, (3, 3), padding='same', strides=(1, 1), dilation_rate=1, kernel_initializer='he_uniform')(inputs)
    if batch_norm:
        conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv2D(init_nb, (3, 3), padding='same', strides=(1, 1), dilation_rate=1, kernel_initializer='he_uniform')(conv1)
    if batch_norm:
        conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv3 = Conv2D(init_nb, (3, 3), padding='same', strides=(2, 2), kernel_initializer='he_uniform')(conv2)
    if batch_norm:
        conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    
    conv4 = Conv2D(init_nb * 2, (3, 3), padding='same', strides=(1, 1), dilation_rate=1, kernel_initializer='he_uniform')(conv3)
    if batch_norm:
        conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv5 = Conv2D(init_nb * 2, (3, 3), padding='same', strides=(1, 1), dilation_rate=1, kernel_initializer='he_uniform')(conv4)
    if batch_norm:
        conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv6 = Conv2D(init_nb * 2, (3, 3), padding='same', strides=(2, 2), kernel_initializer='he_uniform')(conv5)
    if batch_norm:
        conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    conv7 = Conv2D(init_nb * 4, (3, 3), padding='same', strides=(1, 1), dilation_rate=1, kernel_initializer='he_uniform')(conv6)
    if batch_norm:
        conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv8 = Conv2D(init_nb * 4, (3, 3), padding='same', strides=(1, 1), dilation_rate=1, kernel_initializer='he_uniform')(conv7)
    if batch_norm:
        conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv9 = Conv2D(init_nb * 4, (3, 3), padding='same', strides=(2, 2), kernel_initializer='he_uniform')(conv8)
    if batch_norm:
        conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    
    # stacked dilated convolution
    conv10 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=1, kernel_initializer='he_uniform')(conv9)
    if batch_norm:
        conv10 = BatchNormalization()(conv10)
    conv10 = Activation('relu')(conv10)
    conv11 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=2, kernel_initializer='he_uniform')(conv10)
    if batch_norm:
        conv11 = BatchNormalization()(conv11)
    conv11 = Activation('relu')(conv11)
    conv12 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=4, kernel_initializer='he_uniform')(conv11)
    if batch_norm:
        conv12 = BatchNormalization()(conv12)
    conv12 = Activation('relu')(conv12)
    conv13 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=8, kernel_initializer='he_uniform')(conv12)
    if batch_norm:
        conv13 = BatchNormalization()(conv13)
    conv13 = Activation('relu')(conv13)
    conv14 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=16, kernel_initializer='he_uniform')(conv13)
    if batch_norm:
        conv14 = BatchNormalization()(conv14)
    conv14 = Activation('relu')(conv14)
    conv15 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=32, kernel_initializer='he_uniform')(conv14)
    if batch_norm:
        conv15 = BatchNormalization()(conv15)
    conv15 = Activation('relu')(conv15)
    
    dilate_all_added = add([conv10, conv11, conv12, conv13, conv14, conv15])
    
    conv16 = Conv2DTranspose(init_nb * 4, (3, 3), padding='same', strides=(2, 2), kernel_initializer='he_uniform')(dilate_all_added)
    if batch_norm:
        conv16 = BatchNormalization()(conv16)
    conv16 = Activation('relu')(conv16)
    #conv16 = special_concatenate_crop(conv7, conv16, 1)
    conv17 = Conv2D(init_nb * 4, (3, 3), padding='same', kernel_initializer='he_uniform')(conv16)
    if batch_norm:
        conv17 = BatchNormalization()(conv17)
    conv17 = Activation('relu')(conv17)
    
    conv18 = Conv2DTranspose(init_nb * 2, (3, 3), padding='same', strides=(2, 2), kernel_initializer='he_uniform')(conv17)
    if batch_norm:
        conv18 = BatchNormalization()(conv18)
    conv18 = Activation('relu')(conv18)
    conv18 = special_concatenate_pad(conv4, conv18, 0)
    conv19 = Conv2D(init_nb, (3, 3), padding='same', kernel_initializer='he_uniform')(conv18)
    if batch_norm:
        conv19 = BatchNormalization()(conv19)
    conv19 = Activation('relu')(conv19)
    
    conv20 = Conv2DTranspose(init_nb, (3, 3), padding='same', strides=(2, 2), kernel_initializer='he_uniform')(conv19)
    if batch_norm:
        conv20 = BatchNormalization()(conv20)
    conv20 = Activation('relu')(conv20)
    conv20 = special_concatenate_crop(conv1, conv20, 0)
    conv21 = Conv2D(init_nb, (3, 3), padding='same', kernel_initializer='he_uniform')(conv20)
    if batch_norm:
        conv21 = BatchNormalization()(conv21)
    conv21 = Activation('relu')(conv21)
    
    conv22 = Conv2D(num_classes, (1, 1), strides=(1, 1), activation=activation, kernel_initializer='he_uniform')(conv21)

    model = Model(inputs=inputs, outputs=conv22, name=name)

    return model

def map2map_worldNormals(input_shape=(256, 256, 3), init_nb=16, num_classes=1):
    
    inputs = Input(input_shape)
    
    conv1 = Conv2D(init_nb, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv2D(init_nb, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv3 = Conv2D(init_nb, (3, 3), padding='same', strides=(2, 2))(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    
    conv4 = Conv2D(init_nb * 2, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv5 = Conv2D(init_nb * 2, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv6 = Conv2D(init_nb * 2, (3, 3), padding='same', strides=(2, 2))(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    conv7 = Conv2D(init_nb * 4, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(conv6)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv8 = Conv2D(init_nb * 4, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(conv7)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv9 = Conv2D(init_nb * 4, (3, 3), padding='same', strides=(2, 2))(conv8)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    
    # stacked dilated convolution
    conv10 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=1)(conv9)
    conv10 = BatchNormalization()(conv10)
    conv10 = Activation('relu')(conv10)
    conv11 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=2)(conv10)
    conv11 = BatchNormalization()(conv11)
    conv11 = Activation('relu')(conv11)
    conv12 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=4)(conv11)
    conv12 = BatchNormalization()(conv12)
    conv12 = Activation('relu')(conv12)
    conv13 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=8)(conv12)
    conv13 = BatchNormalization()(conv13)
    conv13 = Activation('relu')(conv13)
    conv14 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=16)(conv13)
    conv14 = BatchNormalization()(conv14)
    conv14 = Activation('relu')(conv14)
    conv15 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=32)(conv14)
    conv15 = BatchNormalization()(conv15)
    conv15 = Activation('relu')(conv15)
    
    dilate_all_added = add([conv10, conv11, conv12, conv13, conv14, conv15])
    
    conv16 = Conv2DTranspose(init_nb * 4, (3, 3), padding='same', strides=(2, 2))(dilate_all_added)
    conv16 = BatchNormalization()(conv16)
    conv16 = Activation('relu')(conv16)
    conv16 = concatenate([conv7, conv16])
    conv17 = Conv2D(init_nb * 4, (3, 3), padding='same')(conv16)
    conv17 = BatchNormalization()(conv17)
    conv17 = Activation('relu')(conv17)
    
    conv18 = Conv2DTranspose(init_nb * 2, (3, 3), padding='same', strides=(2, 2))(conv17)
    conv18 = BatchNormalization()(conv18)
    conv18 = Activation('relu')(conv18)
    conv18 = concatenate([conv4, conv18])
    conv19 = Conv2D(init_nb, (3, 3), padding='same')(conv18)
    conv19 = BatchNormalization()(conv19)
    conv19 = Activation('relu')(conv19)
    
    conv20 = Conv2DTranspose(init_nb, (3, 3), padding='same', strides=(2, 2))(conv19)
    conv20 = BatchNormalization()(conv20)
    conv20 = Activation('relu')(conv20)
    conv20 = concatenate([conv1, conv20])
    conv21 = Conv2D(init_nb, (3, 3), padding='same')(conv20)
    conv21 = BatchNormalization()(conv21)
    conv21 = Activation('relu')(conv21)
    
    conv22 = Conv2D(num_classes, (1, 1))(conv21)

    model = Model(inputs=inputs, outputs=conv22)

    return model
    
def map2map_worldNormals_without_BN(input_shape=(256, 256, 3), init_nb=16, num_classes=1):
    
    inputs = Input(input_shape)
    
    conv1 = Conv2D(init_nb, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(inputs)
    #conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv2D(init_nb, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(conv1)
    #conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv3 = Conv2D(init_nb, (3, 3), padding='same', strides=(2, 2))(conv2)
    #conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    
    conv4 = Conv2D(init_nb * 2, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(conv3)
    #conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv5 = Conv2D(init_nb * 2, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(conv4)
    #conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv6 = Conv2D(init_nb * 2, (3, 3), padding='same', strides=(2, 2))(conv5)
    #conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    conv7 = Conv2D(init_nb * 4, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(conv6)
    #conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv8 = Conv2D(init_nb * 4, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(conv7)
    #conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv9 = Conv2D(init_nb * 4, (3, 3), padding='same', strides=(2, 2))(conv8)
    #conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    
    # stacked dilated convolution
    conv10 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=1)(conv9)
    #conv10 = BatchNormalization()(conv10)
    conv10 = Activation('relu')(conv10)
    conv11 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=2)(conv10)
    #conv11 = BatchNormalization()(conv11)
    conv11 = Activation('relu')(conv11)
    conv12 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=4)(conv11)
    #conv12 = BatchNormalization()(conv12)
    conv12 = Activation('relu')(conv12)
    conv13 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=8)(conv12)
    #conv13 = BatchNormalization()(conv13)
    conv13 = Activation('relu')(conv13)
    conv14 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=16)(conv13)
    #conv14 = BatchNormalization()(conv14)
    conv14 = Activation('relu')(conv14)
    conv15 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=32)(conv14)
    #conv15 = BatchNormalization()(conv15)
    conv15 = Activation('relu')(conv15)
    
    dilate_all_added = add([conv10, conv11, conv12, conv13, conv14, conv15])
    
    conv16 = Conv2DTranspose(init_nb * 4, (3, 3), padding='same', strides=(2, 2))(dilate_all_added)
    #conv16 = BatchNormalization()(conv16)
    conv16 = Activation('relu')(conv16)
    conv16 = concatenate([conv7, conv16])
    conv17 = Conv2D(init_nb * 4, (3, 3), padding='same')(conv16)
    #conv17 = BatchNormalization()(conv17)
    conv17 = Activation('relu')(conv17)
    
    conv18 = Conv2DTranspose(init_nb * 2, (3, 3), padding='same', strides=(2, 2))(conv17)
    #conv18 = BatchNormalization()(conv18)
    conv18 = Activation('relu')(conv18)
    conv18 = concatenate([conv4, conv18])
    conv19 = Conv2D(init_nb, (3, 3), padding='same')(conv18)
    #conv19 = BatchNormalization()(conv19)
    conv19 = Activation('relu')(conv19)
    
    conv20 = Conv2DTranspose(init_nb, (3, 3), padding='same', strides=(2, 2))(conv19)
    #conv20 = BatchNormalization()(conv20)
    conv20 = Activation('relu')(conv20)
    conv20 = concatenate([conv1, conv20])
    conv21 = Conv2D(init_nb, (3, 3), padding='same')(conv20)
    #conv21 = BatchNormalization()(conv21)
    conv21 = Activation('relu')(conv21)
    
    conv22 = Conv2D(num_classes, (1, 1))(conv21)

    model = Model(inputs=inputs, outputs=conv22)

    return model


def map2map_semanticSegmentation(input_shape=(256, 256, 3), init_nb=16, num_classes=1):
    
    inputs = Input(input_shape)
    
    conv1 = Conv2D(init_nb, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv2D(init_nb, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv3 = Conv2D(init_nb, (3, 3), padding='same', strides=(2, 2))(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    
    conv4 = Conv2D(init_nb * 2, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv5 = Conv2D(init_nb * 2, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv6 = Conv2D(init_nb * 2, (3, 3), padding='same', strides=(2, 2))(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    conv7 = Conv2D(init_nb * 4, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(conv6)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv8 = Conv2D(init_nb * 4, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(conv7)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv9 = Conv2D(init_nb * 4, (3, 3), padding='same', strides=(2, 2))(conv8)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    
    # stacked dilated convolution
    conv10 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=1)(conv9)
    conv10 = BatchNormalization()(conv10)
    conv10 = Activation('relu')(conv10)
    conv11 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=2)(conv10)
    conv11 = BatchNormalization()(conv11)
    conv11 = Activation('relu')(conv11)
    conv12 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=4)(conv11)
    conv12 = BatchNormalization()(conv12)
    conv12 = Activation('relu')(conv12)
    conv13 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=8)(conv12)
    conv13 = BatchNormalization()(conv13)
    conv13 = Activation('relu')(conv13)
    conv14 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=16)(conv13)
    conv14 = BatchNormalization()(conv14)
    conv14 = Activation('relu')(conv14)
    conv15 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=32)(conv14)
    conv15 = BatchNormalization()(conv15)
    conv15 = Activation('relu')(conv15)
    
    dilate_all_added = add([conv10, conv11, conv12, conv13, conv14, conv15])
    
    conv16 = Conv2DTranspose(init_nb * 4, (3, 3), padding='same', strides=(2, 2))(dilate_all_added)
    conv16 = BatchNormalization()(conv16)
    conv16 = Activation('relu')(conv16)
    conv16 = concatenate([conv7, conv16])
    conv17 = Conv2D(init_nb * 4, (3, 3), padding='same')(conv16)
    conv17 = BatchNormalization()(conv17)
    conv17 = Activation('relu')(conv17)
    
    conv18 = Conv2DTranspose(init_nb * 2, (3, 3), padding='same', strides=(2, 2))(conv17)
    conv18 = BatchNormalization()(conv18)
    conv18 = Activation('relu')(conv18)
    conv18 = concatenate([conv4, conv18])
    conv19 = Conv2D(init_nb * 2, (3, 3), padding='same')(conv18)
    conv19 = BatchNormalization()(conv19)
    conv19 = Activation('relu')(conv19)
    
    conv20 = Conv2DTranspose(init_nb, (3, 3), padding='same', strides=(2, 2))(conv19)
    conv20 = BatchNormalization()(conv20)
    conv20 = Activation('relu')(conv20)
    conv20 = concatenate([conv1, conv20])
    conv21 = Conv2D(init_nb, (3, 3), padding='same')(conv20)
    conv21 = BatchNormalization()(conv21)
    conv21 = Activation('relu')(conv21)
    
    conv22 = Conv2D(num_classes, (1, 1), activation='softmax')(conv21)

    model = Model(inputs=inputs, outputs=conv22)

    return model



def map2map_worldNormals_without_BN_checked_by_mihai(input_shape=(256, 256, 3), init_nb=16, num_classes=1):
    inputs = Input(input_shape)
    
    conv1 = Conv2D(init_nb, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(inputs)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv2D(init_nb, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(conv1)
    conv2 = Activation('relu')(conv2)
    conv3 = Conv2D(init_nb, (3, 3), padding='same', strides=(2, 2))(conv2)
    conv3 = Activation('relu')(conv3)
    
    conv4 = Conv2D(init_nb * 2, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(conv3)
    conv4 = Activation('relu')(conv4)
    conv5 = Conv2D(init_nb * 2, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(conv4)
    conv5 = Activation('relu')(conv5)
    conv6 = Conv2D(init_nb * 2, (3, 3), padding='same', strides=(2, 2))(conv5)
    conv6 = Activation('relu')(conv6)

    conv7 = Conv2D(init_nb * 4, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(conv6)
    conv7 = Activation('relu')(conv7)
    conv8 = Conv2D(init_nb * 4, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(conv7)
    conv8 = Activation('relu')(conv8)
    conv9 = Conv2D(init_nb * 4, (3, 3), padding='same', strides=(2, 2))(conv8)
    conv9 = Activation('relu')(conv9)
    
    # # stacked dilated convolution
    conv10 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=1)(conv9)
    conv10 = Activation('relu')(conv10)
    conv11 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=2)(conv10)
    conv11 = Activation('relu')(conv11)
    conv12 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=4)(conv11)
    conv12 = Activation('relu')(conv12)
    conv13 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=8)(conv12)
    conv13 = Activation('relu')(conv13)
    conv14 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=16)(conv13)
    conv14 = Activation('relu')(conv14)
    conv15 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=32)(conv14)
    conv15 = Activation('relu')(conv15)
    
    dilate_all_added = add([conv10, conv11, conv12, conv13, conv14, conv15])
    
    conv16 = Conv2DTranspose(init_nb * 4, (3, 3), padding='same', strides=(2, 2))(dilate_all_added)
    conv16 = Activation('relu')(conv16)
    conv16 = concatenate([conv7, conv16])
    conv17 = Conv2D(init_nb * 4, (3, 3), padding='same')(conv16)
    conv17 = Activation('relu')(conv17)
    
    conv18 = Conv2DTranspose(init_nb * 2, (3, 3), padding='same', strides=(2, 2))(conv17)
    conv18 = Activation('relu')(conv18)
    conv18 = concatenate([conv4, conv18])
    conv19 = Conv2D(init_nb, (3, 3), padding='same')(conv18)
    conv19 = Activation('relu')(conv19)
    
    conv20 = Conv2DTranspose(init_nb, (3, 3), padding='same', strides=(2, 2))(conv19)
    conv20 = Activation('relu')(conv20)
    conv20 = concatenate([conv1, conv20])
    conv21 = Conv2D(init_nb, (3, 3), padding='same')(conv20)
    conv21 = Activation('relu')(conv21)
    
    conv22 = Conv2D(num_classes, (1, 1), strides=(1, 1))(conv21)

    model = Model(inputs=inputs, outputs=conv22)

    return model

def map2map_worldNormals_checked_by_mihai(input_shape=(256, 256, 3), init_nb=16, num_classes=1):
    
    inputs = Input(input_shape)
    
    conv1 = Conv2D(init_nb, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv2D(init_nb, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv3 = Conv2D(init_nb, (3, 3), padding='same', strides=(2, 2))(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    
    conv4 = Conv2D(init_nb * 2, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv5 = Conv2D(init_nb * 2, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv6 = Conv2D(init_nb * 2, (3, 3), padding='same', strides=(2, 2))(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    conv7 = Conv2D(init_nb * 4, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(conv6)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv8 = Conv2D(init_nb * 4, (3, 3), padding='same', strides=(1, 1), dilation_rate=1)(conv7)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv9 = Conv2D(init_nb * 4, (3, 3), padding='same', strides=(2, 2))(conv8)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    
    # stacked dilated convolution
    conv10 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=1)(conv9)
    conv10 = BatchNormalization()(conv10)
    conv10 = Activation('relu')(conv10)
    conv11 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=2)(conv10)
    conv11 = BatchNormalization()(conv11)
    conv11 = Activation('relu')(conv11)
    conv12 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=4)(conv11)
    conv12 = BatchNormalization()(conv12)
    conv12 = Activation('relu')(conv12)
    conv13 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=8)(conv12)
    conv13 = BatchNormalization()(conv13)
    conv13 = Activation('relu')(conv13)
    conv14 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=16)(conv13)
    conv14 = BatchNormalization()(conv14)
    conv14 = Activation('relu')(conv14)
    conv15 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=32)(conv14)
    conv15 = BatchNormalization()(conv15)
    conv15 = Activation('relu')(conv15)
    
    dilate_all_added = add([conv10, conv11, conv12, conv13, conv14, conv15])
    
    conv16 = Conv2DTranspose(init_nb * 4, (3, 3), padding='same', strides=(2, 2))(dilate_all_added)
    conv16 = BatchNormalization()(conv16)
    conv16 = Activation('relu')(conv16)
    conv16 = concatenate([conv7, conv16])
    conv17 = Conv2D(init_nb * 4, (3, 3), padding='same')(conv16)
    conv17 = BatchNormalization()(conv17)
    conv17 = Activation('relu')(conv17)
    
    conv18 = Conv2DTranspose(init_nb * 2, (3, 3), padding='same', strides=(2, 2))(conv17)
    conv18 = BatchNormalization()(conv18)
    conv18 = Activation('relu')(conv18)
    conv18 = concatenate([conv4, conv18])
    conv19 = Conv2D(init_nb, (3, 3), padding='same')(conv18)
    conv19 = BatchNormalization()(conv19)
    conv19 = Activation('relu')(conv19)
    
    conv20 = Conv2DTranspose(init_nb, (3, 3), padding='same', strides=(2, 2))(conv19)
    conv20 = BatchNormalization()(conv20)
    conv20 = Activation('relu')(conv20)
    conv20 = concatenate([conv1, conv20])
    conv21 = Conv2D(init_nb, (3, 3), padding='same')(conv20)
    conv21 = BatchNormalization()(conv21)
    conv21 = Activation('relu')(conv21)
    
    conv22 = Conv2D(num_classes, (1, 1), strides=(1, 1))(conv21)

    model = Model(inputs=inputs, outputs=conv22)

    return model
