import tensorflow as tf
from tensorflow import app
from tensorflow import flags
import data_utils.reader as reader
import os
import models_baseline
import numpy as np
import math
import time 



def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))



def get_numeric_feature(value):
    '''
    Returns a float or a int64 features depending on the input value type
    :param value:
    '''
    if isinstance(value, (float, np.float)):
        return float_feature(value)
    elif isinstance(value, (int, np.integer)):
        return int64_feature(value)


def build_tf_example(pretrained_data, video_name, class_id, num_frames, num_dims):
    
    prefix = FLAGS.pretrained_model + '/' 
    features = {
            prefix + 'data': bytes_feature(pretrained_data),
            prefix + 'name': bytes_feature(video_name),
            prefix + 'label': get_numeric_feature(class_id),
            prefix + 'num_frames' : get_numeric_feature(num_frames),
            prefix + 'num_dims' : get_numeric_feature(num_dims),
            }

    
    return tf.train.Example(features = tf.train.Features(feature=features))


def print_vars(var_list):
    names = [v.name for v in var_list]
    for name in names:
        print(name)


def build_queue(dir,num_frames, batch_size = 50, num_epochs = 5,crop_height = 64, crop_width = 64, frame_is_random = False):

    filenames = []


    for file in os.listdir(dir):
        if file.endswith('.tfrecords'):
            filenames.append(os.path.join(dir,file))
    print(len(filenames))
    print(filenames) 
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(filenames, num_epochs = num_epochs)
    #print(filename_queue.names)
    
    if (FLAGS.test_data):
        rand_crop_flag = False
        rand_flip_flag = False
    else:
        rand_crop_flag = True
        rand_flip_flag = True

    video, label, name = reader.read_and_decode(filename_queue,  batch_size,crop_height, crop_width,  frame_is_random=False, rand_crop = rand_crop_flag, rand_flip = rand_flip_flag, get_name = True, num_frames = num_frames) #TODO 
    video_batch, label_batch, name_batch = tf.train.shuffle_batch([video, label, name],
    batch_size=batch_size,
    num_threads=2,
    capacity=1000 + 7 * batch_size,
    min_after_dequeue=1000,
    allow_smaller_final_batch = True)

    return video_batch, label_batch, name_batch





def build_graph(video_batch,batch_size,reuse):
    """

    Builds the tensorflow graph for training the model

    """
    model = models_baseline.avg_cnn() #CHANGE MODEL HERE TODO:specify a function for getting this
   
         
    
    features,vars_to_restore = model.build_model(video_batch,batch_size = batch_size,height = FLAGS.crop_height, width = FLAGS.crop_width, num_classes = FLAGS.num_classes,num_frames = FLAGS.num_frames_sampled, ckpt_path = FLAGS.checkpoints_dir,dropout_keep_prob = FLAGS.dropout_keep_prob, reuse=reuse)
    
    return features,vars_to_restore



def evaluate(model_name = 'lstm_convnet' ):
    """
    Evaluates specified model


    """
   
    #optimizer_function = tf.train.AdamOptimizer(FLAGS.learning_rate) 
    
    #gpu_list = [0,1,2]
    vid_features = []
    with tf.variable_scope("graph_variables"):
        video_batch_all, label_batch_all, name_batch_all = build_queue(FLAGS.evaldata_dir,crop_height = FLAGS.crop_height, crop_width = FLAGS.crop_width,  batch_size = FLAGS.num_gpus*FLAGS.batch_size,num_frames = FLAGS.num_frames_sampled, num_epochs = FLAGS.num_epochs)
        
        for i in range(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % ("towernum_", i)) as scope:                        
                    print("building graph for tower: "+str (i))
                    with tf.name_scope("input_data"+str(i)):
                        video_batch = tf.slice(video_batch_all, [i*FLAGS.batch_size,0,0,0,0],[FLAGS.batch_size,-1,-1,-1,-1])
                        label_batch = tf.slice(label_batch_all, [i*FLAGS.batch_size], [FLAGS.batch_size])#reader.build_queu
                        
                    vid_features_tower,vars_to_restore = build_graph(video_batch,FLAGS.batch_size, reuse = tf.get_variable_scope().reuse) 
                    print('shape of feature tower')
                    vid_features_tower = tf.stack(vid_features_tower, axis = 1)
                    print(vid_features_tower.get_shape())
                    vid_features.append(vid_features_tower)
                    tf.get_variable_scope().reuse_variables()
       
    video_pretrained_net_features = tf.concat(vid_features, axis = 0)
    print('all batch pretrained shape')
    print(video_pretrained_net_features.get_shape())
    load_freq = FLAGS.load_freq
    vars_to_load = tf.trainable_variables()
    print('VARS TO LOAD:')
    print_vars(vars_to_load)
    loader = tf.train.Saver(vars_to_load)#vars_to_restore)
   
    
    summary_op = tf.summary.merge_all()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
    
    init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    
    shard_num = 0
    shard_size = int((FLAGS.num_gpus*FLAGS.batch_size)*(math.floor(FLAGS.shard_size/(FLAGS.num_gpus*FLAGS.batch_size))))
    iters_per_shard = int(shard_size/(FLAGS.num_gpus*FLAGS.batch_size))
    print('Shard Size '+str(shard_size))
    print('Iters per Shard ' + str(iters_per_shard))
    MAX_STEPS = FLAGS.max_steps
    try:        
        while not coord.should_stop():                      
            for i in range(MAX_STEPS):               
                if((i==0)):
                   ckpt_name = '/home/anmol/projects/kinetics-baseline/ckpt_11/-16155'#'inception_resnet_pretrained.ckpt'#'./ckpt_11/-16155' #tf.train.latest_checkpoint(FLAGS.checkpoints_dir)
                   print('loading recent checkpoint, from path...' + str(ckpt_name))
                   loader.restore(sess, ckpt_name)
                   print('model-loaded')
                   
                #print("global_step " + str(i))              
                print('shard number '+ str(shard_num))
                
                tf_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.shard_dir,FLAGS.run_name + '_'+ str(shard_num) + '.tfrecords' ))
                for w in range(iters_per_shard): 
                    features, labels, names = sess.run([video_pretrained_net_features,label_batch_all, name_batch_all])
                    if(shard_num==0 and w==0):
                        print('important shapes')
                        print(features.dtype)
                        print(names.dtype)
                        print(labels.dtype)
                        time.sleep(8)
                        #print(features.shape)
                        #print(features[0].shape)
                    #print(labels.shape)
                    #print(names.shape)
                    
                    assert features.shape[0] == int(FLAGS.num_gpus*FLAGS.batch_size)


                    for j in range(features.shape[0]):
                        #print('sample num: '+str(j))
                        [num_frames, num_dims] = features[j].shape #num_frames x dimensionality of layer
                        assert num_frames == FLAGS.num_frames_sampled
                        assert num_dims == 1536
                        '''
                        if(shard_num==0):
                            print(names)
                            print(type(names[0]))
                            print('num frames' + str(num_frames))
                            print('num_dims' + str(num_dims))
                        '''
                        
                        ex = build_tf_example(features[j].tobytes(), names[j], labels[j], num_frames, num_dims)
                        tf_writer.write(ex.SerializeToString())
                shard_num+=1  
                tf_writer.close()

    except tf.errors.OutOfRangeError:
        print('limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


def run():
    tf.set_random_seed(1)
    evaluate()



if (__name__ == "__main__"):

    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('run_name', 'r1', 'prefix of all files')
    flags.DEFINE_string('pretrained_model', 'inception_resnet', 'name of pretrained model')
    flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    flags.DEFINE_integer('max_steps', 100000, 'Number of steps to run trainer.')
    flags.DEFINE_integer('batch_size', 25, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
    
    flags.DEFINE_string('evaldata_dir', '/mnt/AIDATA/anmol/ucf_tfrecords_01/train_tfrecords/', 'Directory to read the tfrecord data.') 
    flags.DEFINE_string('checkpoints_dir','./ckpt_11/', 'Directory of checkpoints to be loaded')
    flags.DEFINE_string('log_dir','./logs_11_test2/','Directory where summaries will be stored')
    flags.DEFINE_float('dropout_keep_prob', 0.5, 'prob of keeping dropout') 
    flags.DEFINE_integer("num_epochs", 5000, "How many passes to make over the dataset before halting training.")
    flags.DEFINE_integer("num_gpus", 1, 'Number of GPUs used')
    flags.DEFINE_integer("load_freq", 300, 'Frequency at which pretrained model is loaded')
    flags.DEFINE_integer("crop_height", 224, "cropped height")
    flags.DEFINE_integer("crop_width", 224, "cropped width")
    flags.DEFINE_integer("num_classes", 101, 'number of classes for classification')
    flags.DEFINE_string('model_name', 'lstm_convnet', 'Model to use')
    flags.DEFINE_string('loss_function', 'softmax_cross_entropy_with_logits', 'name of the loss function used')
    flags.DEFINE_string('optimizer_function', 'AdamOptimizer', 'name of the optimizer used')
    flags.DEFINE_string('job_name', 'local', 'Type of job; local or distributed')
    flags.DEFINE_string('shard_dir', '/mnt/AIDATA/anmol/ucf_pretrained/incresnet', 'directory of pretrained path(not actual shard size btw)')
    flags.DEFINE_integer('num_frames_sampled', 25, 'number of frames sampled from video')

    flags.DEFINE_bool('is_training', 'False', 'is it training?')
    flags.DEFINE_integer('shard_size', 1000, 'num examples in one shard')
    flags.DEFINE_bool('test_data', True, 'is conversion being done on test data')

    print(FLAGS.learning_rate)
    run()



