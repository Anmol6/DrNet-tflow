import tensorflow as tf
import layers as L


def pose_discriminator_tflow(x, name_scope = 'C'):
     
    l = tf.layers.dense(x, 100, activation=tf.nn.relu, name='l1')
    l = tf.layers.dense(l, 100, activation=tf.nn.relu, name='l2')
    l = tf.layers.dense(l, 1, activation=tf.nn.sigmoid, name='l3')
    #l1 = L.fully_connected(x, name='Dense1', activation_fn=tf.nn.relu, n_out = 100)
    #l2 = L.fully_connected(l1, name = 'Dense2',activation_fn = tf.nn.relu, n_out = 100)
    #out = L.fully_connected(l2,name='Dense3',n_out = 1,activation_fn=tf.nn.sigmoid)
    

    return l


