# _*_ coding: utf-8 _*_

import tensorflow as tf

from keras.layers import Conv2D
from keras.layers import UpSampling2D
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import concatenate
from keras.layers import add
from keras import backend as K


"""
Network Components
"""
######## _*_ Convolution + BN _*_ ######## 
def Conv(x, outsize, kernel_size, activation=True):

    out = Conv2D(outsize, kernel_size, strides=1, padding='same', kernel_initializer='he_normal', activation=None)(x)
    out = BatchNormalization(epsilon=1e-5, momentum=0.1)(out)
    if activation:
        out = Activation('relu')(out)
    
    return out


######## _*_ Interactive conv module _*_ ########
def icm(input_tensor, out_channel, kernel_sizes):
    
    # halve the out_channel
    each_channel = int(out_channel / 2)
    
    # do interactive convolution with two different kernel_size
    branch1_1 = Conv(input_tensor, each_channel, (1, kernel_sizes[0]), activation=False)
    branch1_1 = Conv(branch1_1, each_channel, (kernel_sizes[0], 1), activation=False)
    branch2_1 = Conv(input_tensor, each_channel, (1, kernel_sizes[1]), activation=False)
    branch2_1 = Conv(branch2_1, each_channel, (kernel_sizes[1], 1), activation=False)
    
    element_wise_add = add([branch1_1, branch2_1])
    relu = Activation('relu')(element_wise_add)
    
    branch1_2 = Conv(relu, each_channel, (1, kernel_sizes[0]), activation=True)
    branch1_2 = Conv(branch1_2, each_channel, (kernel_sizes[0], 1), activation=True)
    branch2_2 = Conv(relu, each_channel, (1, kernel_sizes[1]), activation=True)
    branch2_2 = Conv(branch2_2, each_channel, (kernel_sizes[1], 1), activation=True)
    
    concate_out = concatenate([branch1_2, branch2_2])
    
    return concate_out

    
######## _*_ Pre Pyramid module _*_ ########
def ppm(input_tensor, out_channel, scales):
    
    scale_in = K.int_shape(input_tensor)[1]
    path_channel = int(out_channel / 2)
    pyramid_channel = int(path_channel / 4)
    
    conv = Conv(input_tensor, path_channel, (1, 1), activation=False)
    pyramid_maps = []
    for scale in scales:
        if scale == scale_in:
            continue
        if scale < scale_in:
            factor = int(scale_in / scale)
            x = MaxPooling2D(pool_size=(factor, factor))(conv)
            x = Conv(x, pyramid_channel, (1, 1), activation=False)
            x = UpSampling2D(size=(factor, factor))(x)
            pyramid_maps.append(x)
        if scale > scale_in:
            factor = int(scale / scale_in)
            x = UpSampling2D(size=(factor, factor))(conv)
            x = Conv(x, pyramid_channel, (1, 1), activation=False)
            x = MaxPooling2D(pool_size=(factor, factor))(x)
            pyramid_maps.append(x)
    x = conv
    for pm in pyramid_maps:
        x = concatenate([x, pm])
    pyramid_out = Conv(x, out_channel, (3, 3), activation=True) 
    
    return pyramid_out


"""
encoder decoder
"""
# ICPN: encoder and decoder
######## _*_ encoder1 _*_ ########
def encoder1(input_tensor, out_channel, kernel_sizes, scales):
    
    identity_inputs = Conv(input_tensor, out_channel, (1, 1), activation=False)
    
    # pre pyramid block
    ppm_ = ppm(input_tensor, out_channel, scales)
    
    # stage1
    icm1 = icm(input_tensor, out_channel, kernel_sizes)
    add1 = add([icm1, identity_inputs])
    relu1 = Activation('relu')(add1)
    
    # stage2
    icm2 = icm(relu1, out_channel, kernel_sizes)
    add2 = add([icm2, relu1, ppm_])
    relu2 = Activation('relu')(add2)
    
    encoder_out = relu2
    
    return encoder_out

######## _*_ decoder1 _*_ ########
def decoder1(input_tensor, features, out_channel, kernel_sizes):
    decoder_out = icm(input_tensor, out_channel, kernel_sizes)
    decoder_out = icm(decoder_out, out_channel, kernel_sizes)

    return decoder_out 

# IcmNet: don not use pyramid structure
######## _*_ encoder2 _*_ ########
def encoder2(input_tensor, out_channel, kernel_sizes, scales):
    identity_inputs = Conv(input_tensor, out_channel, (1, 1), activation=False)
    
    # stage1
    icm1 = icm(input_tensor, out_channel, kernel_sizes)
    add1 = add([icm1, identity_inputs])
    relu1 = Activation('relu')(add1)

    # stage2
    icm2 = icm(relu1, out_channel, kernel_sizes)
    add2 = add([icm2, relu1])
    relu2 = Activation('relu')(add2)

    encoder_out = relu2
    
    return encoder_out

######## _*_ decoder2 _*_ ########
def decoder2(input_tensor, features, out_channel, kernel_sizes):
    decoder_out = icm(input_tensor, out_channel, kernel_sizes)
    decoder_out = icm(decoder_out, out_channel, kernel_sizes)

    return decoder_out 