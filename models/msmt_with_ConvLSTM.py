from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization, add, concatenate, Lambda, Flatten, Conv2DTranspose, ConvLSTM2D, Reshape
from keras.optimizers import RMSprop

import keras.backend as K
import tensorflow as tf

from keras.utils import multi_gpu_model


import numpy as np

from models.losses import bce_dice_loss, dice_coeff

def get_unet_MDCB_with_deconv_layers_with_ConvLSTM(num_sequence_samples, input_height, input_width, num_channels, init_nb=64, lr=0.0001, loss=bce_dice_loss, num_classes=1):
    
    inputs = Input(shape=(num_sequence_samples, input_height, input_width, num_channels), name='Input_layer_0')
    
    #inputs_to_conv_dims = (inputs.shape[1].value, inputs.shape[2].value, inputs.shape[3].value)
    #inputs_reshaped = Reshape(target_shape=inputs_to_conv_dims, name='reshapeinputtoconv')
    
    down1 = ConvLSTM2D(filters=init_nb, kernel_size=(3, 3)
                       , data_format='channels_last'
                       , recurrent_activation='hard_sigmoid'
                       , activation='relu'
                       , padding='same', dilation_rate = 1, return_sequences=True, name='ConvLSTM2D_layer_1')(inputs)
                       
    print('down1')
    print(down1.shape)
    
    down1_reshaped = Reshape(target_shape=(down1.shape[2].value, down1.shape[3].value, down1.shape[4].value), name='reshapeinputtoconv')(down1)
    
    print('down1_reshaped')
    print(down1_reshaped.shape)
    
    down1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same', name='Conv2D_layer_2')(down1_reshaped)
    down1pool = Conv2D(init_nb, (3, 3), activation='relu', padding='same', strides=(2, 2), name='Conv2D_layer_3')(down1)
    #down1pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    
    down2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same', name='Conv2D_layer_4')(down1pool)
    down2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same', name='Conv2D_layer_5')(down2)
    down2pool = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same', strides=(2, 2), name='Conv2D_layer_6')(down2)
    #down2pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)

    down3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same', name='Conv2D_layer_7')(down2pool)
    down3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same', name='Conv2D_layer_8')(down3)
    down3pool = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same', strides=(2, 2), name='Conv2D_layer_9')(down3)
    #down3pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    
    print('down3pool')
    print(down3pool.shape)
    
    stop
    
    # stacked dilated convolution
    down3pool_reshaped = Reshape(target_shape=(1, down3pool.shape[1].value, down3pool.shape[2].value, down3pool.shape[3].value), name='reshapeconvtolstm')(down3pool)
    
    dilate1 = ConvLSTM2D(filters=init_nb*8, kernel_size=(3, 3)
                       , data_format='channels_last'
                       , recurrent_activation='hard_sigmoid'
                       , activation='relu'
                       , padding='same', dilation_rate = 1, return_sequences=True, name='ConvLSTM2D_layer_10')(down3pool_reshaped)
    dilate2 = ConvLSTM2D(filters=init_nb*8, kernel_size=(3, 3)
                       , data_format='channels_last'
                       , recurrent_activation='hard_sigmoid'
                       , activation='relu'
                       , padding='same', dilation_rate = 2, return_sequences=True, name='ConvLSTM2D_layer_11')(dilate1)
    dilate3 = ConvLSTM2D(filters=init_nb*8, kernel_size=(3, 3)
                       , data_format='channels_last'
                       , recurrent_activation='hard_sigmoid'
                       , activation='relu'
                       , padding='same', dilation_rate = 4, return_sequences=True, name='ConvLSTM2D_layer_12')(dilate2)
    dilate4 = ConvLSTM2D(filters=init_nb*8, kernel_size=(3, 3)
                       , data_format='channels_last'
                       , recurrent_activation='hard_sigmoid'
                       , activation='relu'
                       , padding='same', dilation_rate = 8, return_sequences=True, name='ConvLSTM2D_layer_13')(dilate3)
    
    #dilate5 = Conv2D(init_nb*8, (3, 3), activation='relu', padding='same', dilation_rate=16)(dilate4)
    #dilate6 = Conv2D(init_nb*8, (3, 3), activation='relu', padding='same', dilation_rate=32)(dilate5)
    
    dilate_all_added = add([dilate1, dilate2, dilate3, dilate4], name='ADD_layer_14')
    
    dilate_all_added_reshaped = Reshape(target_shape=(dilate_all_added.shape[2].value, dilate_all_added.shape[3].value, dilate_all_added.shape[4].value), name='reshapelstmtoconv')(dilate_all_added)
    
    #up3 = UpSampling2D((2, 2))(dilate_all_added)
    #up3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(up3)
    up3 = Conv2DTranspose(init_nb*4, (3, 3), activation='relu', padding='same', strides=(2, 2), name='Conv2DTranspose_layer_15')(dilate_all_added_reshaped)
    up3 = concatenate([down3, up3], name='Concatenate_layer_16')
    up3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same', name='Conv2D_layer_17')(up3)
    up3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same', name='Conv2D_layer_18')(up3)

    #up2 = UpSampling2D((2, 2))(up3)
    #up2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(up2)
    up2 = Conv2DTranspose(init_nb*2, (3, 3), activation='relu', padding='same', strides=(2, 2), name='Conv2DTranspose_layer_19')(up3)
    up2 = concatenate([down2, up2], name='Concatenate_layer_20')
    up2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same', name='Conv2D_layer_21')(up2)
    up2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same', name='Conv2D_layer_22')(up2)
    
    #up1 = UpSampling2D((2, 2))(up2)
    #up1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(up1)
    up1 = Conv2DTranspose(init_nb, (3, 3), activation='relu', padding='same', strides=(2, 2), name='Conv2DTranspose_layer_23')(up2)
    up1 = concatenate([down1, up1], name='Concatenate_layer_24')
    up1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same', name='Conv2D_layer_25')(up1)
    up1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same', name='Conv2D_layer_26')(up1)
    
    classify = Conv2D(num_classes, (1, 1), activation='sigmoid', name='Conv2D_layer_27')(up1)
    
    model = Model(inputs=inputs, outputs=classify, name='MSMT-Stage-1-TransposeConvs')

    model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])

    return model
    
def get_unet_MDCB_with_deconv_layers_with_ConvLSTM_bottleneck_no_seq(input_height, input_width, num_channels, init_nb=64, lr=0.0001, loss=bce_dice_loss, num_classes=1):
    
    inputs = Input(shape=(input_height, input_width, num_channels), name='Input_layer_0')
    
    down1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same', name='Conv2D_layer_1')(inputs)
    down1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same', name='Conv2D_layer_2')(down1)
    down1pool = Conv2D(init_nb, (3, 3), activation='relu', padding='same', strides=(2, 2), name='Conv2D_layer_3')(down1)
    #down1pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    
    down2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same', name='Conv2D_layer_4')(down1pool)
    down2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same', name='Conv2D_layer_5')(down2)
    down2pool = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same', strides=(2, 2), name='Conv2D_layer_6')(down2)
    #down2pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)

    down3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same', name='Conv2D_layer_7')(down2pool)
    down3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same', name='Conv2D_layer_8')(down3)
    down3pool = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same', strides=(2, 2), name='Conv2D_layer_9')(down3)
    #down3pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    
    # stacked dilated convolution
    down3pool_reshaped = Reshape(target_shape=(1, down3pool.shape[1].value, down3pool.shape[2].value, down3pool.shape[3].value), name='reshapeconvtolstm')(down3pool)
    
    dilate1 = ConvLSTM2D(filters=init_nb*8, kernel_size=(3, 3)
                       , data_format='channels_last'
                       , recurrent_activation='hard_sigmoid'
                       , activation='relu'
                       , padding='same', dilation_rate = 1, return_sequences=True, name='ConvLSTM2D_layer_10')(down3pool_reshaped)
    dilate2 = ConvLSTM2D(filters=init_nb*8, kernel_size=(3, 3)
                       , data_format='channels_last'
                       , recurrent_activation='hard_sigmoid'
                       , activation='relu'
                       , padding='same', dilation_rate = 2, return_sequences=True, name='ConvLSTM2D_layer_11')(dilate1)
    dilate3 = ConvLSTM2D(filters=init_nb*8, kernel_size=(3, 3)
                       , data_format='channels_last'
                       , recurrent_activation='hard_sigmoid'
                       , activation='relu'
                       , padding='same', dilation_rate = 4, return_sequences=True, name='ConvLSTM2D_layer_12')(dilate2)
    dilate4 = ConvLSTM2D(filters=init_nb*8, kernel_size=(3, 3)
                       , data_format='channels_last'
                       , recurrent_activation='hard_sigmoid'
                       , activation='relu'
                       , padding='same', dilation_rate = 8, return_sequences=True, name='ConvLSTM2D_layer_13')(dilate3)
    dilate5 = ConvLSTM2D(filters=init_nb*8, kernel_size=(3, 3)
                       , data_format='channels_last'
                       , recurrent_activation='hard_sigmoid'
                       , activation='relu'
                       , padding='same', dilation_rate = 16, return_sequences=True, name='ConvLSTM2D_layer_14')(dilate4)
    dilate6 = ConvLSTM2D(filters=init_nb*8, kernel_size=(3, 3)
                       , data_format='channels_last'
                       , recurrent_activation='hard_sigmoid'
                       , activation='relu'
                       , padding='same', dilation_rate = 32, return_sequences=True, name='ConvLSTM2D_layer_15')(dilate5)
    
    dilate_all_added = add([dilate1, dilate2, dilate3, dilate4, dilate5, dilate6], name='ADD_layer_16')
    
    dilate_all_added_reshaped = Reshape(target_shape=(dilate_all_added.shape[2].value, dilate_all_added.shape[3].value, dilate_all_added.shape[4].value), name='reshapelstmtoconv')(dilate_all_added)
    
    #up3 = UpSampling2D((2, 2))(dilate_all_added)
    #up3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(up3)
    up3 = Conv2DTranspose(init_nb*4, (3, 3), activation='relu', padding='same', strides=(2, 2), name='Conv2DTranspose_layer_17')(dilate_all_added_reshaped)
    up3 = concatenate([down3, up3], name='Concatenate_layer_18')
    up3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same', name='Conv2D_layer_19')(up3)
    up3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same', name='Conv2D_layer_20')(up3)

    #up2 = UpSampling2D((2, 2))(up3)
    #up2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(up2)
    up2 = Conv2DTranspose(init_nb*2, (3, 3), activation='relu', padding='same', strides=(2, 2), name='Conv2DTranspose_layer_21')(up3)
    up2 = concatenate([down2, up2], name='Concatenate_layer_22')
    up2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same', name='Conv2D_layer_23')(up2)
    up2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same', name='Conv2D_layer_24')(up2)
    
    #up1 = UpSampling2D((2, 2))(up2)
    #up1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(up1)
    up1 = Conv2DTranspose(init_nb, (3, 3), activation='relu', padding='same', strides=(2, 2), name='Conv2DTranspose_layer_25')(up2)
    up1 = concatenate([down1, up1], name='Concatenate_layer_26')
    up1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same', name='Conv2D_layer_27')(up1)
    up1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same', name='Conv2D_layer_28')(up1)
    
    classify = Conv2D(num_classes, (1, 1), activation='sigmoid', name='Conv2D_layer_29')(up1)
    
    model = Model(inputs=inputs, outputs=classify, name='ConvLSTM_no_seq')

    model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])

    return model
    
    
def get_unet_MDCB_with_deconv_layers_only_with_ConvLSTM(num_sequence_samples, input_height, input_width, num_channels, init_nb=64, lr=0.0001, loss=bce_dice_loss, num_classes=1):
    
    inputs = Input(shape=(num_sequence_samples, input_height, input_width, num_channels), name='Input_layer_0')
    
    #inputs_to_conv_dims = (inputs.shape[1].value, inputs.shape[2].value, inputs.shape[3].value)
    #inputs_reshaped = Reshape(target_shape=inputs_to_conv_dims, name='reshapeinputtoconv')
    
    down1 = ConvLSTM2D(filters=init_nb, kernel_size=(3, 3)
                       , data_format='channels_last'
                       , recurrent_activation='hard_sigmoid'
                       , activation='relu'
                       , padding='same', dilation_rate = 1, return_sequences=True, name='ConvLSTM2D_layer_1')(inputs)
                       
    print('down1')
    print(down1.shape)
    
    down1_reshaped = Reshape(target_shape=(down1.shape[2].value, down1.shape[3].value, down1.shape[4].value), name='reshapeinputtoconv')(down1)
    
    print('down1_reshaped')
    print(down1_reshaped.shape)
    
    down1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same', name='Conv2D_layer_2')(down1_reshaped)
    down1pool = Conv2D(init_nb, (3, 3), activation='relu', padding='same', strides=(2, 2), name='Conv2D_layer_3')(down1)
    #down1pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    
    down2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same', name='Conv2D_layer_4')(down1pool)
    down2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same', name='Conv2D_layer_5')(down2)
    down2pool = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same', strides=(2, 2), name='Conv2D_layer_6')(down2)
    #down2pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)

    down3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same', name='Conv2D_layer_7')(down2pool)
    down3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same', name='Conv2D_layer_8')(down3)
    down3pool = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same', strides=(2, 2), name='Conv2D_layer_9')(down3)
    #down3pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    
    print('down3pool')
    print(down3pool.shape)
    
    stop
    
    # stacked dilated convolution
    down3pool_reshaped = Reshape(target_shape=(1, down3pool.shape[1].value, down3pool.shape[2].value, down3pool.shape[3].value), name='reshapeconvtolstm')(down3pool)
    
    dilate1 = ConvLSTM2D(filters=init_nb*8, kernel_size=(3, 3)
                       , data_format='channels_last'
                       , recurrent_activation='hard_sigmoid'
                       , activation='relu'
                       , padding='same', dilation_rate = 1, return_sequences=True, name='ConvLSTM2D_layer_10')(down3pool_reshaped)
    dilate2 = ConvLSTM2D(filters=init_nb*8, kernel_size=(3, 3)
                       , data_format='channels_last'
                       , recurrent_activation='hard_sigmoid'
                       , activation='relu'
                       , padding='same', dilation_rate = 2, return_sequences=True, name='ConvLSTM2D_layer_11')(dilate1)
    dilate3 = ConvLSTM2D(filters=init_nb*8, kernel_size=(3, 3)
                       , data_format='channels_last'
                       , recurrent_activation='hard_sigmoid'
                       , activation='relu'
                       , padding='same', dilation_rate = 4, return_sequences=True, name='ConvLSTM2D_layer_12')(dilate2)
    dilate4 = ConvLSTM2D(filters=init_nb*8, kernel_size=(3, 3)
                       , data_format='channels_last'
                       , recurrent_activation='hard_sigmoid'
                       , activation='relu'
                       , padding='same', dilation_rate = 8, return_sequences=True, name='ConvLSTM2D_layer_13')(dilate3)
    
    #dilate5 = Conv2D(init_nb*8, (3, 3), activation='relu', padding='same', dilation_rate=16)(dilate4)
    #dilate6 = Conv2D(init_nb*8, (3, 3), activation='relu', padding='same', dilation_rate=32)(dilate5)
    
    dilate_all_added = add([dilate1, dilate2, dilate3, dilate4], name='ADD_layer_14')
    
    dilate_all_added_reshaped = Reshape(target_shape=(dilate_all_added.shape[2].value, dilate_all_added.shape[3].value, dilate_all_added.shape[4].value), name='reshapelstmtoconv')(dilate_all_added)
    
    #up3 = UpSampling2D((2, 2))(dilate_all_added)
    #up3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(up3)
    up3 = Conv2DTranspose(init_nb*4, (3, 3), activation='relu', padding='same', strides=(2, 2), name='Conv2DTranspose_layer_15')(dilate_all_added_reshaped)
    up3 = concatenate([down3, up3], name='Concatenate_layer_16')
    up3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same', name='Conv2D_layer_17')(up3)
    up3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same', name='Conv2D_layer_18')(up3)

    #up2 = UpSampling2D((2, 2))(up3)
    #up2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(up2)
    up2 = Conv2DTranspose(init_nb*2, (3, 3), activation='relu', padding='same', strides=(2, 2), name='Conv2DTranspose_layer_19')(up3)
    up2 = concatenate([down2, up2], name='Concatenate_layer_20')
    up2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same', name='Conv2D_layer_21')(up2)
    up2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same', name='Conv2D_layer_22')(up2)
    
    #up1 = UpSampling2D((2, 2))(up2)
    #up1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(up1)
    up1 = Conv2DTranspose(init_nb, (3, 3), activation='relu', padding='same', strides=(2, 2), name='Conv2DTranspose_layer_23')(up2)
    up1 = concatenate([down1, up1], name='Concatenate_layer_24')
    up1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same', name='Conv2D_layer_25')(up1)
    up1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same', name='Conv2D_layer_26')(up1)
    
    classify = Conv2D(num_classes, (1, 1), activation='sigmoid', name='Conv2D_layer_27')(up1)
    
    model = Model(inputs=inputs, outputs=classify, name='MSMT-Stage-1-TransposeConvs')

    model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])

    return model
    
def get_unet_MDCB_with_deconv_layers_without_ConvLSTM_same_params(input_shape=(1024,1024,3), init_nb=64, lr=0.0001, loss=bce_dice_loss, num_classes=1):
    
    inputs = Input(input_shape, name='Input_layer_0')
    
    down1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same', name='Conv2D_layer_1')(inputs)
    down1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same', name='Conv2D_layer_2')(down1)
    down1pool = Conv2D(init_nb, (3, 3), activation='relu', padding='same', strides=(2, 2), name='Conv2D_layer_3')(down1)
    #down1pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    
    down2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same', name='Conv2D_layer_4')(down1pool)
    down2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same', name='Conv2D_layer_5')(down2)
    down2pool = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same', strides=(2, 2), name='Conv2D_layer_6')(down2)
    #down2pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)

    down3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same', name='Conv2D_layer_7')(down2pool)
    down3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same', name='Conv2D_layer_8')(down3)
    down3pool = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same', strides=(2, 2), name='Conv2D_layer_9')(down3)
    #down3pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    
    # stacked dilated convolution
    
    dilate1 = Conv2D(filters=init_nb*8, kernel_size=(3, 3)
                       , activation='relu'
                       , padding='same', dilation_rate = 1, name='Conv2D_layer_10')(down3pool)
    dilate2 = Conv2D(filters=init_nb*8, kernel_size=(3, 3)
                       , activation='relu'
                       , padding='same', dilation_rate = 2, name='Conv2D_layer_11')(dilate1)
    dilate3 = Conv2D(filters=init_nb*8, kernel_size=(3, 3)
                       , activation='relu'
                       , padding='same', dilation_rate = 4, name='Conv2D_layer_12')(dilate2)
    dilate4 = Conv2D(filters=init_nb*8, kernel_size=(3, 3)
                       , activation='relu'
                       , padding='same', dilation_rate = 8, name='Conv2D_layer_13')(dilate3)
    
    #dilate5 = Conv2D(init_nb*8, (3, 3), activation='relu', padding='same', dilation_rate=16)(dilate4)
    #dilate6 = Conv2D(init_nb*8, (3, 3), activation='relu', padding='same', dilation_rate=32)(dilate5)
    
    dilate_all_added = add([dilate1, dilate2, dilate3, dilate4], name='ADD_layer_14')
    
    #up3 = UpSampling2D((2, 2))(dilate_all_added)
    #up3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(up3)
    up3 = Conv2DTranspose(init_nb*4, (3, 3), activation='relu', padding='same', strides=(2, 2), name='Conv2DTranspose_layer_15')(dilate_all_added)
    up3 = concatenate([down3, up3], name='Concatenate_layer_16')
    up3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same', name='Conv2D_layer_17')(up3)
    up3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same', name='Conv2D_layer_18')(up3)

    #up2 = UpSampling2D((2, 2))(up3)
    #up2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(up2)
    up2 = Conv2DTranspose(init_nb*2, (3, 3), activation='relu', padding='same', strides=(2, 2), name='Conv2DTranspose_layer_19')(up3)
    up2 = concatenate([down2, up2], name='Concatenate_layer_20')
    up2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same', name='Conv2D_layer_21')(up2)
    up2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same', name='Conv2D_layer_22')(up2)
    
    #up1 = UpSampling2D((2, 2))(up2)
    #up1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(up1)
    up1 = Conv2DTranspose(init_nb, (3, 3), activation='relu', padding='same', strides=(2, 2), name='Conv2DTranspose_layer_23')(up2)
    up1 = concatenate([down1, up1], name='Concatenate_layer_24')
    up1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same', name='Conv2D_layer_25')(up1)
    up1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same', name='Conv2D_layer_26')(up1)
    
    classify = Conv2D(num_classes, (1, 1), activation='sigmoid', name='Conv2D_layer_27')(up1)
    
    model = Model(inputs=inputs, outputs=classify, name='MSMT-Stage-1-TransposeConvs')

    model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])

    return model

