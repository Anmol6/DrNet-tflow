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
import models.resnet_v2 as rv2
from models.resnet18 import resnet_18 
def lrelu(x, leak=0.2):
     #with tf.variable_scope(name):
     #f1 = 0.5 * (1 + leak)
     #f2 = 0.5 * (1 - leak)
     
     #return f1 * x + f2 * abs(x)
     return tf.maximum(x, x*leak)
def conv_conv_pool(input_, n_filters, training, name, filter_dims = (3,3),pool=True, pad_mode = 'same',activation=tf.nn.relu,pool_square_size = 2, custom_activation=False):
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
            net = tf.layers.conv2d(net, F, filter_dims, activation=None, padding=pad_mode, name="conv_{}".format(i + 1))
            net = tf.layers.batch_normalization(net, training=training, name="bn_{}".format(i + 1))
            if (custom_activation==True):
                net = activation(net, name="relu{}_{}".format(name, i + 1))
            else:
                net=lrelu(net)
        if pool is False:
            return net
        psz = pool_square_size 
        pool = tf.layers.max_pooling2d(net, (psz, psz), strides=(psz, psz), name="pool_{}".format(name))

        return net, pool



def upsample_concat(inputA, input_B, name, size = (2,2),reuse=True):
    """Upsample `inputA` and concat with `input_B`

    Args:
        input_A (4-D Tensor): (N, H, W, C)
        input_B (4-D Tensor): (N, 2*H, 2*H, C2)
        name (str): name of the concat operation

    Returns:
        output (4-D Tensor): (N, 2*H, 2*W, C + C2)
    """
    upsample = upsampling_2D(inputA, size=size, reuse=reuse,name=name)
    if (input_B == None):
        return upsample
    return tf.concat([upsample, input_B], axis=-1, name="concat_{}".format(name))


def upsampling_2D(tensor, name, reuse,size=(2, 2)):
    """Upsample/Rescale `tensor` by size

    Args:
        tensor (4-D Tensor): (N, H, W, C)
        name (str): name of upsampling operations
        size (tuple, optional): (height_multiplier, width_multiplier)
            (default: (2, 2))

    Returns:
        output (4-D Tensor): (N, h_multiplier * H, w_multiplier * W, C)
    """
    H, W, n_in = tensor.get_shape().as_list()[1:]

    H_multi, W_multi = size
    target_H = H * H_multi
    target_W = W * W_multi

    return tf.layers.conv2d_transpose(tensor, n_in, (3,3), strides=(2,2), name='upsampled_'+str(H)+'x'+str(W), reuse=reuse,padding='SAME')#tf.image.resize_nearest_neighbor(tensor, (target_H, target_W), name="upsample_{}".format(name))


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
        
        with tf.variable_scope(Ep_scope,reuse = reuse_Ep):    
            '''
            out = resnet_18(x, self._size_embedding_p, is_train = training)#resnet_v2.resnet_v2_50(x, num_classes = self._size_embedding_p, reuse=reuse_Ep)
            out = tf.nn.tanh(out)
            
            print('Ep shape: ')
            print(out.get_shape())
            '''
            #conv1_ec, pool1_ec = conv_conv_pool(x, [64], training, name=1) #128 -> 64
            conv2_ec, pool2_ec = conv_conv_pool(x, [64], training, name=2) #64 -> 32
            conv3_ec, pool3_ec = conv_conv_pool(pool2_ec, [128], training, name=3) #16 -> 8
            conv4_ec, pool4_ec = conv_conv_pool(pool3_ec, [256], training, name=4)#8 -> 4
            conv5_ec, pool5_ec = conv_conv_pool(pool4_ec, [512], training, name = 5)#4 -> 1
                        
            conv6_ec = conv_conv_pool(pool5_ec, [self._size_embedding_p],training,filter_dims = (4,4), custom_activation = True, pool=False,pad_mode = 'valid',activation=tf.nn.tanh, name=6)
            
            #import ipdb
            #ipdb.set_trace()
            out = tf.squeeze(conv6_ec)
        
        return out


    def Ec(self, x, Ec_scope, reuse_Ec, training = True, make_Unet = False):     
        
        with tf.variable_scope(Ec_scope, reuse = reuse_Ec):
                       
            #conv1_ec, pool1_ec = conv_conv_pool(x, [64], training, name=1) #128 -> 64
            conv2_ec, pool2_ec = conv_conv_pool(x, [64], training, name=2) #64 -> 32
            conv3_ec, pool3_ec = conv_conv_pool(pool2_ec, [128], training, name=3) #16 -> 8
            conv4_ec, pool4_ec = conv_conv_pool(pool3_ec, [256], training, name=4)#8 -> 4
            conv5_ec, pool5_ec = conv_conv_pool(pool4_ec, [512], training, name = 5)#4 -> 2
                        
            conv6_ec = conv_conv_pool(pool5_ec, [self._size_embedding_c],training,filter_dims = (4,4), custom_activation = True, pool=False,pad_mode = 'valid',activation=tf.nn.tanh, name=6)
            
            print('Content Encoding Shape')
            print(conv6_ec.get_shape())
        
        if(make_Unet):
            return conv6_ec, [conv5_ec, conv4_ec, conv3_ec,conv2_ec]
        
        return conv6_ec    
    
    def D_hp_given(self, x_ec, Ec_scope, reuse_Ec,h_Ep, D_scope, reuse_D, training=True, n_channels = 3):
            
            h_Ec, Ulayers = self.Ec(x_ec, Ec_scope,reuse_Ec, training=training,make_Unet = True) 
            h_Ep = tf.expand_dims(tf.expand_dims(h_Ep,axis=1),axis=1)
                       
            #ipdb.set_trace()
            #h_Ep_ = tf.layers.conv2d_transpose(h_Ep, self._size_embedding_p,(1,1), (128, 128), name="upsample_{}".format('lol'), reuse=reuse_D)
            with tf.variable_scope(D_scope, reuse = reuse_D): 
                encoding_content_and_pose = tf.concat([h_Ep, h_Ec], axis= -1)
                #import ipdb;
                #ipdb.set_trace()

                #TODO: maybe change dimensionality of encoding_content_and_pose first?
                
                up8 = tf.layers.conv2d_transpose(encoding_content_and_pose, 512,(4,4),name = '8')
                #up8 = upsampling_2D(encoding_content_and_pose,name= 8,size = (4,4))  #1->4
                #import ipdb
                #ipdb.set_trace()
                up9 = upsample_concat(up8, None, reuse=reuse_D, name=9) # 4 -> 8 
                conv9 = conv_conv_pool(up9, [512], training, name=9, pool=False)

                up10 = upsample_concat(conv9, Ulayers[1], reuse=reuse_D, name=10) #8->16
                conv10 = conv_conv_pool(up10, [256], training, name=10, pool=False)

                up11 = upsample_concat(conv10, Ulayers[2], reuse=reuse_D, name=11) #16->32
                conv11 = conv_conv_pool(up11, [128], training, name=11, pool=False)
                
                
                up12 = upsample_concat(conv11, Ulayers[3], reuse=reuse_D, name=12) #32->64
                conv12 = conv_conv_pool(up12, [64], training, name=12, pool=False)

                #up13 = upsample_concat(conv12, Ulayers[4], reuse=reuse_D, name=13) #64->128
                #conv13 = conv_conv_pool(up13, [64], training, name=13, pool=False)

                
                out = tf.layers.conv2d(conv12, n_channels, (3, 3), name='final', activation=tf.nn.sigmoid, padding='same')       
                print('output shape')
                print(out.get_shape())

            return out, h_Ec#[encoding_content_and_pose, conv13, h_Ep, h_Ec]


