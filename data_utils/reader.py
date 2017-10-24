import tensorflow as tf 
from keras import backend as K
import numpy as np 
import matplotlib.pyplot as plt
import os
import math
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def read_and_decode(filename_queue, batch_size,crop_height, crop_width, num_frames, num_channels=3,frame_is_random = False, rand_frame_list = None, center_image = True, resize_image_0 = True, resize_image = False, rand_crop = True, rand_flip = False,is_training = True,resized_small = 256,  resized_height = 224, resized_width = 224, crop_with_pad = False, get_name = False, crop_center = False,train_drnet = True, div_by_255 = True, max_steps = 12, dataset = 'ucf'):
    
    """Preprocessing for  video data 
    :param filename_queue: queue of files to read
    :param crop_height: height of cropped frame
    :param crop_width: width of cropped frame
    :param num_frames: number of frames to sample from each video 
    :param num_channels: number of channels in a video frame, default to 3
    :param frame_is_random: whether frames sampled are random
    :param reduce_mean: whether to subtract mean of all frames from video frames
    :param rand_crop: whether to crop image randomnly. If so, cropped to crop_height x crop_width
    :param size_resized_small: size to which smaller dimension of image is resized, if None image is not resized
    :param resized_height: height to resize image
    :param resized_width: width to resize image
    :param resize_image: whether to resize image
    :param crop_with_pad: whether to crop or pad the image to resized_height x resized_width (without scaling). If False and resize_image is True, image is resized with scaling
    :param max_steps: how far the next frames can be from the first frame, integer
    :param get_name: whether to return name of video
    :param center_crop: whether to do central cropping of image
    :param train_drnet: whether training drnet or lstm (True if training drnet), Boolean
    """



    reader = tf.TFRecordReader()
    feature_dict={
        'video/height': tf.FixedLenFeature([], tf.int64),
        'video/width': tf.FixedLenFeature([], tf.int64),
        'video/num_frames': tf.FixedLenFeature([], tf.int64), 
        'video/encoded': tf.FixedLenFeature([],tf.string), #TODO:CHANGE THIS LATER
        'video/class/label': tf.FixedLenFeature([], tf.int64),
        'video/name': tf.FixedLenFeature([], tf.string)
        }
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features = feature_dict )
    
    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    
    print(type(features['video/encoded']))
    print(type(features['video/name']))
    
    video = tf.decode_raw(features['video/encoded'],tf.uint8)
    name = tf.cast(features['video/name'], tf.string) 
    height = tf.cast(features['video/height'], tf.int32)

    width = tf.cast(features['video/width'], tf.int32)
    num_framestf = tf.cast(features['video/num_frames'], tf.int64)
    label = features['video/class/label']
    
               
    if(dataset == 'kth'):
        h0 = 120 
        w0 = 160
        video = tf.reshape(video, [-1, h0,w0, num_channels])

        
    if (resize_image_0):
        video = tf.cond(height>width, lambda:tf.image.resize_images(video, size = [height*tf.cast(tf.floor(resized_small/width),tf.int32), resized_small]),lambda:tf.image.resize_images(video, size = [resized_small, width*(tf.cast(tf.floor(resized_small/height),tf.int32))]))
        video.set_shape([-1, 256, 341, 3])#TODO: have a function for this
    if(crop_center):
        if(dataset=='ucf'):
            video = tf.image.crop_to_bounding_box(video, 8, 48,224,224)
            
    if frame_is_random:
        if(train_drnet):
            offset_one = tf.random_uniform(shape = [],minval = 0, maxval=num_framestf-max_steps-3,dtype=tf.int64)
            offset_two = tf.random_uniform(shape = [], minval=offset_one+1, maxval=offset_one+max_steps, dtype=tf.int64)
            offset_three = tf.random_uniform(shape = [], minval=offset_one+1, maxval=offset_one+max_steps, dtype=tf.int64)
            offset_four = tf.random_uniform(shape = [], minval=offset_one+1, maxval=offset_one+max_steps, dtype=tf.int64)
            offset_five = tf.random_uniform(shape = [], minval=offset_one+1, maxval=offset_one+max_steps, dtype=tf.int64)

            frame_list = [offset_one, offset_two, offset_three, offset_four, offset_five]
            video = tf.gather(video, indices = frame_list)
            num_frames = 5
             
        else: 
            first_index = tf.random_uniform(shape=[], minval=0, maxval=num_framestf-num_frames-3, dtype=tf.int64)
            frame_list = tf.linspace(tf.to_float(first_index), tf.to_float(first_index+num_frames-1), num_frames)
            video = tf.gather(video, indices = tf.cast(frame_list, tf.int64))
        #video = tf.expand_dims(video, axis=0)
    else:
        slice_indices = tf.linspace(0,num_framestf-1, num_frames,dtype=np.int64)
        video = tf.gather(video, slice_indices)
    #ipdb.set_trace() 
    

    if(rand_crop and is_training):
        video = tf.random_crop(video,size = [num_frames,crop_height, crop_width,num_channels])
        video = tf.reshape(video, [num_frames,crop_height, crop_width,num_channels]) 
    if(rand_flip and is_training):
        flip = tf.random_uniform(shape = [1], minval = 0.0, maxval = 1.0)        
        video =tf.cond(flip[0]>0.5, lambda: tf.reverse(video, axis=[2]),lambda:video)
    if(resize_image and not crop_with_pad):# and is_training):
        video = tf.image.resize_images(video,size = [resized_height, resized_width]) 
    if(resize_image and crop_with_pad):
        video = tf.image.resize_image_with_crop_or_pad(video, resized_height, resized_width)

    if(resize_image): 
        video = tf.reshape(video, [num_frames, resized_height,resized_width, num_channels])
    
        
    if(div_by_255):
        video = tf.divide(tf.to_float(video),tf.constant(255.0))
    
    if (get_name):
        return video, label, name
    return video, label

 

           
             

def build_queue(dir,num_frames, batch_size = 50, num_epochs = 5,crop_height = 64, crop_width = 64):

    filenames = []


    for file in os.listdir(dir):
        if file.endswith('.tfrecords'):
            filenames.append(os.path.join(dir,file))

    print(filenames) 
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(filenames, num_epochs = num_epochs)
    print(filename_queue.names)
    video, label = read_and_decode(filename_queue,batch_size,crop_height, crop_width,num_frames = num_frames) #TODO 
    video_batch, label_batch = K.tf.train.shuffle_batch([video, label],
    batch_size=batch_size,
    num_threads=2,
    capacity=1000 + 3 * batch_size,
    min_after_dequeue=1000)

    return video_batch, label_batch



