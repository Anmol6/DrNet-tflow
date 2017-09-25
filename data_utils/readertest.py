import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import ipdb
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




def read_and_decode(filename_queue, batch_size,crop_height, crop_width, num_frames, num_channels=3,frame_is_random = True, center_image = True, resize_image = True, rand_crop = True, rand_flip = True,is_training = True,  resized_height = 299, resized_width = 299, crop_with_pad = False, dataset = 'ucf'):
    reader = tf.TFRecordReader()
    feature_dict={
        'video/height': tf.FixedLenFeature([], tf.int64),
        'video/width': tf.FixedLenFeature([], tf.int64),
        'video/num_frames': tf.FixedLenFeature([], tf.int64), 
        'video/encoded': tf.FixedLenFeature([],tf.string), #TODO:CHANGE THIS LATER
        'video/class/label': tf.FixedLenFeature([], tf.int64)
        }
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features = feature_dict )
    
    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    print(type(features['video/encoded']))
    video = tf.decode_raw(features['video/encoded'],tf.uint8)
    
    height = tf.cast(features['video/height'], tf.int32)

    width = tf.cast(features['video/width'], tf.int32)
    num_framestf = tf.cast(features['video/num_frames'], tf.int32)
    label = features['video/class/label']
    # Random transformations can be put here: right before you crop images
    # to predefined size. To get more information look at the stackoverflow
    # question linked above.
    #print(type(video))
    if (dataset=='ucf'):
        num_frames_total = 25
        h0 = 341
        w0 = 256
    #num_frames_ucf = 25
    #frame_stride = int(math.floor(num_frames_ucf/(num_frames-1)))
    if (dataset=='ucf'):
        video = tf.reshape(video, [num_frames_total, h0, w0,num_channels])
    #import ipdb; ipdb.set_trace() 
    #print('ORIGINAL SHAPE')
    #print(video.get_shape())
    if frame_is_random:
        rand_frame_index = tf.floor(num_frames_total*tf.random_uniform([1], minval = 0.0, maxval = 1.0))[0]
        rand_frame_index = tf.cast(rand_frame_index, tf.int64)
        video = tf.slice(video, begin = [rand_frame_index,0,0,0],size = [1,-1,-1,-1])#video[rand_frame_index,:,:,:]
        #video = tf.expand_dims(video, axis=0)
    else:
        slice_indices = np.linspace(0,num_frames_total-1, num_frames,dtype=np.int64)
        video = tf.gather(video, slice_indices)
    #ipdb.set_trace() 
    
    
    
    if(center_image):
        mean = tf.constant(np.load('ucfmean4.npy').astype('float32'))
        video = tf.to_float(video) - mean
    if(rand_crop and is_training):
        video = tf.random_crop(video,size = [num_frames,crop_height, crop_width,num_channels])
    if(rand_flip and is_training):
        flip = tf.random_uniform(shape = [1], minval = 0.0, maxval = 1.0)        
        video =tf.cond(flip[0]>0.5, lambda: tf.reverse(video, axis=[2]),lambda:video)
    if(resize_image and not crop_with_pad):# and is_training):
        video = tf.image.resize_images(video,size = [resized_height, resized_width]) 
    if(resize_image and crop_with_pad):
        video = tf.image.resize_image_with_crop_or_pad(video, resized_height, resized_width)

    if(resize_image): 
        video = tf.reshape(video, [num_frames, resized_height,resized_width, num_channels])
   
    #ipdb.set_trace()
    return video, label

def find_mean_and_stdev(tfrecords_dir,batch_size=20, crop_height=224, crop_width=224, num_frames=5):

    filenames = []


    for file in os.listdir(tfrecords_dir):
        if file.endswith('.tfrecords'):
            filenames.append(os.path.join(tfrecords_dir,file))

    #print(filenames) 
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(filenames, num_epochs = 1)
    #print(filename_queue.names)
    print('here')
    video, label = read_and_decode(filename_queue,batch_size,crop_height, crop_width,num_frames = num_frames,frame_is_random = False,resize_image=False,rand_crop=False, rand_flip = False,center_image=False, is_training=True) #TODO 
    video_batch, label_batch = tf.train.shuffle_batch([video, label],
    batch_size=batch_size,
    num_threads=2,
    capacity=1000 + 3 * batch_size,
    min_after_dequeue=1000)
    wo = 299
    ho = 299
    print('here2')
    mean_arr = np.zeros((ho, wo, 3))
    std_arr = np.zeros((ho,wo, 3))
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())

    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    num_batches = 0
    #print('HEIGHT')
    #print(sess.run(height))
    #print(sess.run(width))
    bb = tf.to_float(video_batch)
    #while not coord.should_stop():
    for i in range(100000):
        print('step ' + str(i))
        #bb = tf.to_float(video_batch)
        try:
            vid_np = bb.eval(session = sess)
        except tf.errors.OutOfRangeError:
            coord.request_stop()
            break
             
        mean_batch = np.mean(vid_np,axis=(0,1)) #compute mean along frames and batch
        if(i%100==0):
            print('saving...')
            np.save('ucfmean_19frames.npy', mean_arr)

        mean_arr += mean_batch
        num_batches+=1
        if coord.should_stop():
            break
    mean_arr = mean_arr/num_batches
    print('saving final')
    np.save('ucfmean_19frames.npy', mean_arr)
    print('total batches: ' + str(num_batches))





if (__name__ == "__main__"):
    find_mean_and_stdev('/mnt/AIDATA/anmol/ucf_tfrecords_1/data/ucf101_tfrecord_files_train')  
        
       
