import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.activations import relu
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.layers import (Activation, Add, BatchNormalization, Concatenate,
                          Conv2D, DepthwiseConv2D, Dropout,
                          GlobalAveragePooling2D, Input, Lambda, Reshape,
                          Softmax, ZeroPadding2D)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_file
import cv2
import numpy as np

from Xception import Xception


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation('relu')(x)

    # 首先使用3x3的深度可分离卷积
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    # 利用1x1卷积进行通道数调整
    x = Conv2D(filters, (1, 1), padding='same', use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    return x


def res_zp_Deeplabv3(input_shape=(512, 512, 3), classes=21, alpha=1., OS=16):
    img_input = Input(shape=input_shape)

    # x         32, 32, 2048
    # skip1     128, 128, 256
    x, atrous_rates, skip1 = Xception(img_input, alpha, OS=OS)
    size_before = tf.keras.backend.int_shape(x)
    #--------------------改进代码---------------------------------

    # 调整通道 32, 32, 2048 -> 32, 32, 1024
    c0 = Conv2D(1024, (1, 1), padding='same', use_bias=False, name='image_tune_pooling')(x)
    #print(c0.shape)
    c0 = tf.keras.layers.Reshape((1024, 1024, 1))(c0)
    # print(skip1.shape)
    # print(skip1.dtype)
    # ------------------------------------------------------------
    # ---------------------------------------------------------------#
    #   全部求平均后，再利用expand_dims扩充维度
    #   64,64,2048 -> 1,1,2048 -> 1,1,2048
    #   下面开始并行空洞卷积
    # ---------------------------------------------------------------#
    b4_0 = GlobalAveragePooling2D()(x)
    b4_0 = Lambda(lambda x: K.expand_dims(x, 1))(b4_0)
    b4_0 = Lambda(lambda x: K.expand_dims(x, 1))(b4_0)
    b4_0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='image_pooling_0')(b4_0)
    b4_0 = BatchNormalization(name='image_pooling_BN_0', epsilon=1e-5)(b4_0)
    b4_0 = Activation('relu')(b4_0)
    # 1,1,256 -> 64,64,256
    b4_0 = Lambda(lambda x: tf.image.resize(x, size_before[1:3]))(b4_0)

    # ---------------------------------------------------------------#
    #   调整通道
    # ---------------------------------------------------------------#
    b0_0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0_0')(x)
    b0_0 = BatchNormalization(name='aspp0_BN_0', epsilon=1e-5)(b0_0)
    b0_0 = Activation('relu', name='aspp0_activation_0')(b0_0)

    # ---------------------------------------------------------------#
    #   rate值与OS相关，SepConv_BN为先3x3膨胀卷积，再1x1卷积，进行压缩
    #   其膨胀率就是rate值
    # ---------------------------------------------------------------#
    b1_0 = SepConv_BN(x, 256, 'aspp1_0', rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
    b2_0 = SepConv_BN(x, 256, 'aspp2_0', rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
    b3_0 = SepConv_BN(x, 256, 'aspp3_0', rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

    # 64, 64, 256 + 64, 64, 256 + 64, 64, 256 + 64, 64, 256 + 64, 64, 256 -> 64, 64, 1280
    x1 = Concatenate()([b4_0, b0_0, b1_0, b2_0, b3_0])

    b4 = GlobalAveragePooling2D()(c0)
    #print(b4.shape)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Conv2D(4, (1, 1), padding='same', use_bias=False, name='image_pooling_1')(b4)
    b4 = BatchNormalization(name='image_pooling_BN_1', epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)
    # 1,1,256 -> 64,64,256
    b4 = Lambda(lambda x: tf.image.resize(x, (1024,1024)))(b4)

    # ---------------------------------------------------------------#
    #   调整通道
    # ---------------------------------------------------------------#
    b0 = Conv2D(4, (1, 1), padding='same', use_bias=False, name='aspp0_1')(c0)
    b0 = BatchNormalization(name='aspp0_BN_1', epsilon=1e-5)(b0)
    b0 = Activation('relu', name='aspp0_activation_1')(b0)

    # ---------------------------------------------------------------#
    #   rate值与OS相关，SepConv_BN为先3x3膨胀卷积，再1x1卷积，进行压缩
    #   其膨胀率就是rate值
    # ---------------------------------------------------------------#
    b1 = SepConv_BN(c0, 4, 'aspp1_1', rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
    b2 = SepConv_BN(c0, 4, 'aspp2_1', rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
    b3 = SepConv_BN(c0, 4, 'aspp3_1', rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

    # 64, 64, 256 + 64, 64, 256 + 64, 64, 256 + 64, 64, 256 + 64, 64, 256 -> 64, 64, 1280
    x2 = Concatenate()([b4, b0, b1, b2, b3])
    x2 = Conv2D(1, (1, 1), padding='same',
               use_bias=False, name='concat_projection_0')(x2)
    #print(x2.shape)
    x2 = tf.keras.layers.Reshape((32, 32, 1024))(x2)
    x = Concatenate()([x1, x2])
    # 利用1x1卷积调整通道数
    # 64, 64, 1280 -> 64,64,256
    x = Conv2D(256, (1, 1), padding='same',
               use_bias=False, name='concat_projection')(x)
    # 并行空洞卷积结束
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    # skip1.shape[1:3] 为 128,128
    # 64,64,256 -> 128,128,256
    x = Lambda(lambda xx: tf.image.resize(xx, skip1.shape[1:3]))(x)

    # 128,128,24 -> 128,128,48
    dec_skip1 = Conv2D(48, (1, 1), padding='same', use_bias=False, name='feature_projection0')(skip1)
    dec_skip1 = BatchNormalization(name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    dec_skip1 = Activation('relu')(dec_skip1)

    # 128,128,256 + 128,128,48 -> 128,128,304
    
    x = Concatenate()([x, dec_skip1])
    # 128,128,304 -> 128,128,256 -> 128,128,256
    x = SepConv_BN(x, 256, 'decoder_conv0',
                   depth_activation=True, epsilon=1e-5)
    x = SepConv_BN(x, 256, 'decoder_conv1',
                   depth_activation=True, epsilon=1e-5)

    # 128,128,256 -> 128,128,2 -> 512,512,2
    
    x = Conv2D(classes, (1, 1), padding='same')(x)

    size_before3 = tf.keras.backend.int_shape(img_input)
    x = Lambda(lambda xx: tf.image.resize(xx, size_before3[1:3]))(x)

    x = Reshape((-1, classes))(x)
    x = Softmax()(x)

    inputs = img_input
    model = Model(inputs, x, name='deeplabv3plus')

    return model

