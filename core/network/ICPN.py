# _*_ coding: utf-8 _*_

from keras.models import *
from keras.layers import *
from keras.optimizers import *

from .base_modules.modules import *

# ICPN 
def network(input_size=(256,256,3)):
    
    with tf.variable_scope("Super_Parameters"):
        
        ks = [3, 5]
        scales = [256, 128, 64, 32, 16]
        out_channels = [32, 64, 128, 256, 512]
    
    with tf.variable_scope("Encoder"):
        
        inputs = Input(input_size)
        conv = Conv(inputs, 32, 3, activation=True)
        
        encode1 = encoder1(conv, out_channels[0], ks, scales) # (256, 32)
        pool1 = MaxPooling2D(pool_size=(2, 2))(encode1)
        
        encode2 = encoder1(pool1, out_channels[1], ks, scales) # (128, 64)
        pool2 = MaxPooling2D(pool_size=(2, 2))(encode2)
        
        encode3 = encoder1(pool2, out_channels[2], ks, scales) # (64, 128)
        pool3 = MaxPooling2D(pool_size=(2, 2))(encode3)
        
        encode4 = encoder1(pool3, out_channels[3], ks, scales) # (32, 256)
        pool4 = MaxPooling2D(pool_size=(2, 2))(encode4)
        
        encode5 = encoder1(pool4, out_channels[4], ks, scales) # (16, 512)
        pool5 = MaxPooling2D(pool_size=(2, 2))(encode5)
        
    with tf.variable_scope("Medium_conv"):
        
        medium = Conv(pool5, 1024, 3, activation='relu')
    
    with tf.variable_scope("Decoder"):
        
        up5 = UpSampling2D(size=(2, 2))(medium)
        decode5 = decoder1(up5, encode5, out_channels[4], ks)
        
        up4 = UpSampling2D(size=(2, 2))(decode5)
        decode4 = decoder1(up4, encode4, out_channels[3], ks)
        
        up3 = UpSampling2D(size=(2, 2))(decode4)
        decode3 = decoder1(up3, encode3, out_channels[2], ks)
        
        up2 = UpSampling2D(size=(2, 2))(decode3)
        decode2 = decoder1(up2, encode2, out_channels[1], ks)
        
        up1 = UpSampling2D(size=(2, 2))(decode2)
        decode1 = decoder1(up1, encode1, out_channels[0], ks)
    
    with tf.variable_scope("predict"):
        
        conv_ = Conv(decode1, 32, 3, activation=True)
        conv_ = Conv(conv_, 32, 3, activation=True)
        logits = Conv2D(2, 1, activation = 'sigmoid')(conv_)
    
    model = Model(input = inputs, output=logits)

    return model
