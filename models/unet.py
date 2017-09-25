"""
Simple U-Net implementation in TensorFlow

Objective: detect vehicles

y = f(X)

X: image (640, 960, 3)
y: mask (640, 960, 1)
   - binary image
   - background is masked 0
   - vehicle is masked 255

Loss function: maximize IOU

    (intersection of prediction & grount truth)
    -------------------------------
    (union of prediction & ground truth)

Notes:
    In the paper, the pixel-wise softmax was used.
    But, I used the IOU because the datasets I used are
    not labeled for segmentations

Original Paper:
    https://arxiv.org/abs/1505.04597
"""
import time
import os
import pandas as pd
import tensorflow as tf
import resnet_v2
from resnet18 import resnet_18 

def conv_conv_pool(input_, n_filters, training, name, pool=True, activation=tf.nn.relu,pool_square_size = 2, no_act=False):
    """{Conv -> BN -> RELU}x2 -> {Pool, optional}

    Args:
        input_ (4-D Tensor): (batch_size, H, W, C)
        n_filters (list): number of filters [int, int]
        training (1-D Tensor): Boolean Tensor
        name (str): name postfix
        pool (bool): If True, MaxPool2D
        activation: Activaion functions

    Returns:
        net: output of the Convolution operations
        pool (optional): output of the max pooling operations
    """
    net = input_

    with tf.variable_scope("layer{}".format(name)):
        for i, F in enumerate(n_filters):
            net = tf.layers.conv2d(net, F, (3, 3), activation=None, padding='same', name="conv_{}".format(i + 1))
            net = tf.layers.batch_normalization(net, training=training, name="bn_{}".format(i + 1))
            if (no_act==False):
                net = activation(net, name="relu{}_{}".format(name, i + 1))

        if pool is False:
            return net
        psz = pool_square_size 
        pool = tf.layers.max_pooling2d(net, (psz, psz), strides=(psz, psz), name="pool_{}".format(name))

        return net, pool


def upsample_concat(inputA, input_B, name, size = (2,2)):
    """Upsample `inputA` and concat with `input_B`

    Args:
        input_A (4-D Tensor): (N, H, W, C)
        input_B (4-D Tensor): (N, 2*H, 2*H, C2)
        name (str): name of the concat operation

    Returns:
        output (4-D Tensor): (N, 2*H, 2*W, C + C2)
    """
    upsample = upsampling_2D(inputA, size=size, name=name)

    return tf.concat([upsample, input_B], axis=-1, name="concat_{}".format(name))


def upsampling_2D(tensor, name, size=(2, 2)):
    """Upsample/Rescale `tensor` by size

    Args:
        tensor (4-D Tensor): (N, H, W, C)
        name (str): name of upsampling operations
        size (tuple, optional): (height_multiplier, width_multiplier)
            (default: (2, 2))

    Returns:
        output (4-D Tensor): (N, h_multiplier * H, w_multiplier * W, C)
    """
    H, W, _ = tensor.get_shape().as_list()[1:]

    H_multi, W_multi = size
    target_H = H * H_multi
    target_W = W * W_multi

    return tf.image.resize_nearest_neighbor(tensor, (target_H, target_W), name="upsample_{}".format(name))


def make_unet(X, training,n_channels = 3):
    """Build a U-Net architecture

    Args:
        X (4-D Tensor): (N, H, W, C)
        training (1-D Tensor): Boolean Tensor is required for batchnormalization layers

    Returns:
        output (4-D Tensor): (N, H, W, C)
            Same shape as the `input` tensor

    Notes:
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://arxiv.org/abs/1505.04597
    """
    net = X
    net = tf.layers.conv2d(net, 3, (1, 1), name="colour_space_adjust")
    conv1, pool1 = conv_conv_pool(X, [8, 8], training, name=1)
    conv2, pool2 = conv_conv_pool(pool1, [16, 16], training, name=2)
    conv3, pool3 = conv_conv_pool(pool2, [32, 32], training, name=3)
    conv4, pool4 = conv_conv_pool(pool3, [64, 64], training, name=4)
    conv5 = conv_conv_pool(pool4, [128, 128], training, name=5, pool=False)

    up6 = upsample_concat(conv5, conv4, name=6)
    conv6 = conv_conv_pool(up6, [64, 64], training, name=6, pool=False)

    up7 = upsample_concat(conv6, conv3, name=7)
    conv7 = conv_conv_pool(up7, [32, 32], training, name=7, pool=False)

    up8 = upsample_concat(conv7, conv2, name=8)
    conv8 = conv_conv_pool(up8, [16, 16], training, name=8, pool=False)

    up9 = upsample_concat(conv8, conv1, name=9)
    conv9 = conv_conv_pool(up9, [8, 8], training, name=9, pool=False)

    return tf.layers.conv2d(conv9, n_channels, (1, 1), name='final', activation=tf.nn.sigmoid, padding='same'), [conv1,conv4, pool4, conv5, up9, conv9] 



class Unet_models:

    def __init__(self, size_encoding_p, size_encoding_c):
        self._size_embedding_p = size_encoding_p
        self._size_embedding_c = size_encoding_c
        return 
    def Ep(self,x, Ep_scope, reuse_Ep, training=True):
        
        with tf.variable_scope(Ep_scope,reuse = reuse_Ep ):    
            out = resnet_18(x, self._size_embedding_p, is_train = training)#resnet_v2.resnet_v2_50(x, num_classes = self._size_embedding_p, reuse=reuse_Ep)
            out = tf.nn.sigmoid(out)
            print('Ep shape!!!')
            
            print(out.get_shape())
            
            '''
            conv0_ep, pool0_ep = conv_conv_pool(x, [32, 32], training, name =0)
            conv1_ep, pool1_ep = conv_conv_pool(pool0_ep, [64, 64], training, name=1)
            conv2_ep, pool2_ep = conv_conv_pool(pool1_ep, [64, 64], training, name=2)
            conv3_ep, pool3_ep = conv_conv_pool(pool2_ep, [128, 128], training, name=3)
            conv4_ep, pool4_ep = conv_conv_pool(pool3_ep, [128, 128], training, name=4)
            conv5_ep, pool5_ep = conv_conv_pool(pool4_ep, [256, 256], training, name=5)
            conv6_ep, pool6_ep = conv_conv_pool(pool5_ep, [256, 512], training, name=6)
            conv7_ep, pool7_ep = conv_conv_pool(pool6_ep, [512, 512], training, name=7)
            conv8_ep = conv_conv_pool(pool7_ep, [512, self._size_embedding_p], training, pool = False, name = 8)
            out = conv8_ep
            '''
        return out


    def Ec(self, x, Ec_scope, reuse_Ec, training = True, make_Unet = False):     
        
        with tf.variable_scope(Ec_scope, reuse = reuse_Ec):
            conv0_ec, pool0_ec = conv_conv_pool(x,[32,32],training, name=0) 
            conv1_ec, pool1_ec = conv_conv_pool(pool0_ec, [64, 64], training, name=1)
            conv2_ec, pool2_ec = conv_conv_pool(pool1_ec, [64, 64], training, name=2)
            conv3_ec, pool3_ec = conv_conv_pool(pool2_ec, [128,128], training, name=3)
            conv4_ec, pool4_ec = conv_conv_pool(pool3_ec, [128, 128], training, name=4)
            conv5_ec, pool5_ec = conv_conv_pool(pool4_ec, [256,256], training, name = 5)
            conv6_ec, pool6_ec = conv_conv_pool(pool5_ec, [256, 512], training, name=6)
            print(pool6_ec.get_shape())
            conv7_ec, pool7_ec = conv_conv_pool(pool6_ec, [512, 512], training, name=7)
            conv8_ec = conv_conv_pool(pool7_ec, [512, self._size_embedding_c], activation = tf.nn.sigmoid, training=training, name=8,  pool = False,no_act=False)
        if(make_Unet):
            return conv8_ec, [conv7_ec, conv6_ec, conv5_ec, conv4_ec, conv3_ec, conv2_ec, conv1_ec, conv0_ec]
        
        return conv8_ec    
    
    
    def D(self, x_ec, Ec_scope, reuse_Ec, x_ep, Ep_scope,reuse_Ep,D_scope, reuse_D, training=True, n_channels = 3):
        
        h_Ec, Ulayers = self.Ec(x_ec, Ec_scope,reuse_Ec, make_Unet = True) 
        h_Ep = tf.expand_dims(self.Ep(x_ep, Ep_scope, reuse_Ep),axis=1)
        
        with tf.variable_scope(D_scope, reuse = reuse_D): 
            encoding_content_and_pose = tf.concat([tf.expand_dims(h_Ep,axis=1), h_Ec], axis= -1)
            #TODO: maybe change dimensionality of encoding_content_and_pose first?
            up9 = upsample_concat(encoding_content_and_pose, Ulayers[0], name=9)
            conv9 = conv_conv_pool(up9, [self._size_embedding_c+10, 512], training, name=9, pool=False)

            up10 = upsample_concat(conv9, Ulayers[1], name=10)
            conv10 = conv_conv_pool(up10, [256, 256], training, name=10, pool=False)

            up11 = upsample_concat(conv10, Ulayers[2], name=11)
            conv11 = conv_conv_pool(up11, [256, 128], training, name=11, pool=False)

            up12 = upsample_concat(conv11, Ulayers[3], name=12)
            conv12 = conv_conv_pool(up12, [128, 128], training, name=12, pool=False)

            up13 = upsample_concat(conv12,Ulayers[4] , name=13)
            conv13 = conv_conv_pool(up13, [128, 128], training, name=13, pool=False)

            up14 = upsample_concat(conv13, Ulayers[5], name=14)
            conv14 = conv_conv_pool(up14, [64, 64], training, name=14, pool=False)

            up15 = upsample_concat(conv14, Ulayers[6], name=15)
            conv15 = conv_conv_pool(up15, [64, 64], training, name=15, pool=False)

            up16 = upsample_concat(conv15, Ulayers[7], name=16)
            conv16 = conv_conv_pool(up16, [32, 32], training, name=16, pool=False)

            out = tf.layers.conv2d(conv16, n_channels, (1, 1), name='final', activation=tf.nn.sigmoid, padding='same')       


        return out


