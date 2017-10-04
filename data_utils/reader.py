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
    :param train_drnet: whether to use custom drnet scheme for getting frames, Boolean
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
    # Random transformations can be put here: right before you crop images
    # to predefined size. To get more information look at the stackoverflow
    # question linked above.
    #print(type(video))
        
    '''    
    if (dataset=='ucf'):
        num_frames_total = 25
        h0 = 240
        w0 = 320        
        video = tf.reshape(video, [num_frames_total, h0, w0, num_channels])
    '''
    if(dataset == 'kth'):
        h0 = 120 
        w0 = 160
        video = tf.reshape(video, [-1, h0,w0, num_channels])

    if(center_image and dataset == 'ucf'):
        mean = tf.constant(np.load('data_utils/ucftruemean.npy').astype('float32'))
        video = tf.to_float(video) - mean
   
    
    if (resize_image_0):
        video = tf.cond(height>width, lambda:tf.image.resize_images(video, size = [height*tf.cast(tf.floor(resized_small/width),tf.int32), resized_small]),lambda:tf.image.resize_images(video, size = [resized_small, width*(tf.cast(tf.floor(resized_small/height),tf.int32))]))
        video.set_shape([-1, 256, 341, 3])#TODO: have a function for this
    if(crop_center):
        if(dataset=='ucf'):
            video = tf.image.crop_to_bounding_box(video, 8, 48,224,224)
            
    if frame_is_random:
        if(train_drnet):
            offset_first_image = tf.random_uniform(shape = [],minval = 0, maxval=num_framestf-max_steps-2,dtype=tf.int64)
            #offset_second_image = tf.random_uniform(shape = [],minval = offset_first_image+1, maxval=offset_first_image+max_steps,dtype=tf.int64)
            #offset_third_image = tf.random_uniform(shape = [],minval = offset_first_image+1, maxval=offset_first_image+max_steps,dtype=tf.int64)

            frame_list = tf.linspace(tf.cast(offset_first_image, tf.float32), tf.cast(offset_first_image+max_steps-1, tf.float32), max_steps)
            video = tf.gather(video, indices = tf.cast(frame_list, tf.int64))
            num_frames = max_steps      
        elif (rand_frame_list == None):
            rand_frame_index = tf.floor(num_frames_total*tf.random_uniform([1], minval = 0.0, maxval = 1.0))[0]
            rand_frame_index = tf.cast(rand_frame_index, tf.int64)
            video = tf.slice(video, begin = [rand_frame_index,0,0,0],size = [1,-1,-1,-1])

        else:
            video = tf.gather(video, indices = rand_frame_list)
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
    
    print('Video size')
    print(video.get_shape())
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




def read_tfrecords_file(tfrecords_filename):
	record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
	vids = []
	names = []
	dims = []
	for string_record in record_iterator:
	    
	    example = tf.train.Example()
	    example.ParseFromString(string_record)
	    #print(type(example.features.feature['image/height'].int64_list.value))
	    #print(example.ListFields())


	    height = int(example.features.feature['video/height'].int64_list.value[0])
	    
	    width = int(example.features.feature['video/width'].int64_list.value[0])
	    
	    vid_string = (example.features.feature['video/encoded'].bytes_list.value[0])
	    
	    name_string = (example.features.feature['video/name'].bytes_list.value[0])
	    label_string = (example.features.feature['video/class/label'].bytes_list.value)
	    #print(name_string)
	    #print(label_string)

	    vid_1d = np.fromstring(vid_string, dtype=np.uint8)
	    reconstructed_vid = vid_1d.reshape((-1, height, width, 3))
	    dims.append([height,width])
	    vids.append(reconstructed_vid)
	    names.append(name_string)
	    #vids.append(reconstructed_vid)
	return vids,names,dims

if (__name__ == "__main__"):
	#tfrecord_dir = '/home/anmol/projects/kinetics-baseline/data/ucf101_tfrecord_files/train-00001-of-00095.tfrecords'
        tfrecord_dir = '/mnt/AIDATA/anmol/ucf_tfrecords_01/train_tfrecords/train-00001-of-00095.tfrecords'
        
        vids,names,dims = read_tfrecords_file(tfrecord_dir)
        print('DONE')
        print(np.array(vids).shape)
        print(np.array(dims).shape)
        np.save('vidtestnew.npy', vids)
        np.save('dimsnew.npy', dims)

