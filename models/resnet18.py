from collections import namedtuple

import tensorflow as tf
import numpy as np

import models.resnet_18_utils as utils

def resnet_18(inp, size_enc, is_train = True):
    #print('Building model')
    # filters = [128, 128, 256, 512, 1024]
    filters = [64, 64, 128, 256, 512]
    kernels = [7, 3, 3, 3, 3]
    strides = [2, 0, 2, 2, 2]

    # conv1
    print('\tBuilding unit: conv1')
    with tf.variable_scope('conv1'):
        x = conv(inp, kernels[0], filters[0], strides[0])
        x = bn(x, is_train)
        x = relu(x)
        x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

    # conv2_x
    x = residual_block(x, is_train, name='conv2_1')
    x = residual_block(x, is_train,name='conv2_2')

    # conv3_x
    x = residual_block_first(x, is_train, filters[2], strides[2], name='conv3_1')
    x = residual_block(x,is_train, name='conv3_2')

    # conv4_x
    x = residual_block_first(x, is_train, filters[3], strides[3], name='conv4_1')
    x = residual_block(x, is_train, name='conv4_2')

    # conv5_x
    x = residual_block_first(x, is_train, filters[4], strides[4], name='conv5_1')
    x = residual_block(x, is_train, name='conv5_2')

    # Logit
    with tf.variable_scope('logits') as scope:
        print('\tBuilding unit: %s' % scope.name)
        x = tf.reduce_mean(x, [1, 2])
        x = tf.layers.dense(x,size_enc)#fc(x, size_enc)
    
    logits = x
    return logits


def residual_block_first( x, is_train, out_channel, strides, name="unit"):
    in_channel = x.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        print('\tBuilding residual unit: %s' % scope.name)

        # Shortcut connection
        if in_channel == out_channel:
            if strides == 1:
                shortcut = tf.identity(x)
            else:
                shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
        else:
            shortcut = conv(x, 1, out_channel, strides, name='shortcut')
        # Residual
        x = conv(x, 3, out_channel, strides, name='conv_1')
        x = bn(x, is_train, name='bn_1')
        x = relu(x, name='relu_1')
        x = conv(x, 3, out_channel, 1, name='conv_2')
        x = bn(x, is_train, name='bn_2')
        # Merge
        x = x + shortcut
        x = relu(x, name='relu_2')
    return x


def residual_block( x, is_train, input_q=None, output_q=None, name="unit"):
    num_channel = x.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        print('\tBuilding residual unit: %s' % scope.name)
        # Shortcut connection
        shortcut = x
        # Residual
        x = conv(x, 3, num_channel, 1, input_q=input_q, output_q=output_q, name='conv_1')
        x = bn(x, is_train, name='bn_1')
        x = relu(x, name='relu_1')
        x = conv(x, 3, num_channel, 1, input_q=output_q, output_q=output_q, name='conv_2')
        x = bn(x, is_train, name='bn_2')

        x = x + shortcut
        x = relu(x, name='relu_2')
    return x

def conv( x, filter_size, out_channel, stride, pad="SAME", input_q=None, output_q=None, name="conv"):
    b, h, w, in_channel = x.get_shape().as_list()
    x = utils._conv(x, filter_size, out_channel, stride, pad, input_q, output_q, name)
    f = 2 * (h/stride) * (w/stride) * in_channel * out_channel * filter_size * filter_size
    w = in_channel * out_channel * filter_size * filter_size
    scope_name = tf.get_variable_scope().name + "/" + name
    #add_flops_weights(scope_name, f, w)
    return x

def fc( x, out_dim, input_q=None, output_q=None, name="fc"):
    b, in_dim = x.get_shape().as_list()
    x = utils._fc(x, out_dim, input_q, output_q, name)
    f = 2 * (in_dim + 1) * out_dim
    w = (in_dim + 1) * out_dim
    scope_name = tf.get_variable_scope().name + "/" + name
    #add_flops_weights(scope_name, f, w)
    return x

def bn( x, is_train, name="bn"):
    x=tf.layers.batch_normalization(x, training=is_train, trainable=True, name=name)
    #x = utils._bn(x, is_train,  name)
    # f = 8 * get_data_size(x)
    # w = 4 * x.get_shape().as_list()[-1]
    # scope_name = tf.get_variable_scope().name + "/" + name
    # add_flops_weights(scope_name, f, w)
    return x

def relu( x, name="relu"):
    x = utils._relu(x, 0.0, name)
    # f = get_data_size(x)
    # scope_name = tf.get_variable_scope().name + "/" + name
    # add_flops_weights(scope_name, f, 0)
    return x

def get_data_size( x):
    return np.prod(x.get_shape().as_list()[1:])

def add_flops_weights( scope_name, f, w):
    counted
    if scope_name not in counted_scope:
        flops += f
        weights += w
        counted_scope.append(scope_name)

