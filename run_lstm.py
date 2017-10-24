import tensorflow as tf
import matplotlib.pyplot as plt
#from models.unet2 import Unet_models as Unet_model
from models.pose_encoder import pose_discriminator_tflow as C
from models.unet_dcgan import Unet_models as Unet_model
from models.lstm_model import make_lstm_cell
import numpy as np
import os
import math 
from data_utils import reader
from tensorflow.python.client import device_lib


flags = tf.app.flags
FLAGS = flags.FLAGS


def get_available_gpus():
    """prints available devices
    """
    local_device_protos = device_lib.list_local_devices()
    l= [x.name for x in local_device_protos]
    for name in l:
        print(name)

def bce(o, t):
    """computes binary cross entropy loss
    """
    #o = tf.clip_by_value(o, 1e-7, 1. - 1e-7)
    return -tf.reduce_mean(t*tf.log(o) + (1-t)*tf.log(1-o))


def print_vars(var_list):
    """Prints names of Tensors in var_list
    """
    names = [v.name for v in var_list]
    for name in names:
        print(name)

def train():
    """Trains LSTM on DrNet embeddings

    """
        

    #Data reading pipeline
    filenames = []
    for file in os.listdir(FLAGS.train_data_dir):
        if file.endswith('.tfrecords'):
            filenames.append(os.path.join(FLAGS.train_data_dir,file))
  
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(filenames, num_epochs = FLAGS.num_epochs)
    
    
    #start_index = tf.random_uniform(shape = [],minval = 0, maxval=FLAGS.max_start_index,dtype=tf.int64)
    #frame_list = tf.linspace(tf.to_float(start_index), tf.to_float(start_index + FLAGS.num_steps_total),tf.constant(FLAGS.num_steps_total+1,dtype=tf.int32), name = "frame_list")
    #frame_list = tf.cast(frame_list, tf.int64)
               
    video, label = reader.read_and_decode(filename_queue,FLAGS.batch_size,FLAGS.crop_height, FLAGS.crop_width,num_frames =FLAGS.num_steps_total+1, resized_height = FLAGS.resized_height, resized_width = FLAGS.resized_width,frame_is_random=True, rand_frame_list = None, resize_image = True, crop_with_pad = False, rand_crop = False, resize_image_0 = False,dataset = FLAGS.dataset, train_drnet=False) #TODO 
    
    video_batch, label_batch = tf.train.shuffle_batch([video, label],
    batch_size=int(FLAGS.num_gpus*FLAGS.batch_size),
    num_threads=20,
    capacity=2000 + 3 * FLAGS.batch_size,
    min_after_dequeue=100)
    print('BATCH SHAPE')
    print(video_batch.get_shape())
    video_batch = tf.to_float(video_batch)
    tf.summary.scalar('video_mean', tf.reduce_mean(video_batch))
      
    model = Unet_model(FLAGS.size_pose_embedding, FLAGS.size_content_embedding)
    total_l2_loss_Ep = []
    with tf.variable_scope(FLAGS.lstm_scope):
        cell, state = make_lstm_cell(FLAGS.batch_size, FLAGS.state_size, FLAGS.num_layers, reuse=False)
    
 
    with tf.name_scope('drnet') as scope:    
        for i in range(FLAGS.num_gpus):
            reuse_flag = (i>0)
            with tf.device('/gpu:' + str(i)), tf.name_scope('tower_'+str(i)):
                
               
                video_batch_tower = tf.slice(video_batch, begin = [i*FLAGS.batch_size, 0,0,0,0], size = [FLAGS.batch_size,-1,-1,-1,-1])
                                
                #get pose embedding from first  num_steps_observed frames only
                hp_frames_tower_list = [None]*(FLAGS.num_steps_total+1) #+1 because we need an extra frame due to the offset b/w input, output

                for t in range(FLAGS.num_steps_total+1):
                    frame_t = video_batch_tower[:,t,:,:,:]
                    hp_frames_tower_list[t] = model.Ep(frame_t , FLAGS.Ep_scope, reuse_Ep=(i>0 or t>0),training=FLAGS.training)
                hp_frames_tower = tf.stack(hp_frames_tower_list, axis = 1)
                tf.summary.scalar('hp_frames_observed_tower', tf.reduce_mean(hp_frames_tower))
                
                #get content embedding from first frame only
                frame_first = video_batch_tower[:, 0, :,:,:]               
                hc_frame_first = tf.squeeze(model.Ec(frame_first, FLAGS.Ec_scope, reuse_Ec = reuse_flag, training=FLAGS.training)) 
                tf.summary.scalar('hc_frame_first',tf.reduce_mean(hc_frame_first))
                
                outputs = [None]*FLAGS.num_steps_total
                
                #process input for first num_steps_observed timesteps, start auto-regressing afterwards
                for t in range(FLAGS.num_steps_total):

                    if(t<FLAGS.num_steps_observed):
                        x_in = tf.concat([hp_frames_tower[:,t,:], hc_frame_first], axis = 1)
                        
                    else:
                        assert t>0
                        x_in = tf.concat([outputs[t-1], hc_frame_first],axis=1)
                    
                    out, state = cell(x_in, state)#reuse = (i>0 or t>0))
                    outputs[t] = tf.layers.dense(out, FLAGS.size_pose_embedding, activation = tf.nn.tanh, reuse = (i>0 or t>0), name = FLAGS.dense_scope) 
                #decode output from time num_steps_observed -> num_steps_total
                
                #ipdb.set_trace()
                hp_predictions_all = tf.stack(outputs,axis=1)
                hp_predictions = hp_predictions_all[:,FLAGS.num_steps_observed:,:]
                hp_targets = hp_frames_tower[:,(FLAGS.num_steps_observed+1):,:]
                l2_loss_Ep = tf.losses.mean_squared_error(hp_targets, hp_predictions)
                total_l2_loss_Ep.append(l2_loss_Ep)


                num_predicted = FLAGS.num_steps_total - FLAGS.num_steps_observed
                frame_predictions = [None]*num_predicted
                frame_true = [None]*num_predicted
                for t in range(num_predicted):                    
                    frame_predictions[t], debug_list = model.D_hp_given(frame_first, FLAGS.Ec_scope, True, hp_predictions[:,t,:], FLAGS.D_scope, reuse_D = (i>0 or t>0), training = FLAGS.training)
                    frame_true[t], debug_list = model.D_hp_given(frame_first, FLAGS.Ec_scope, True, hp_targets[:,t,:], FLAGS.D_scope, reuse_D = True, training = FLAGS.training)#only for debugging
                frame_predictions = tf.stack(frame_predictions, axis=1)
                frame_true = tf.stack(frame_true, axis=1)
                
                    
    loss_final = tf.reduce_mean(total_l2_loss_Ep)
    tf.summary.scalar('Ep_l2_loss', loss_final)                
        
    print('lstm variables:')
    lstm_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=FLAGS.lstm_scope)
    print_vars(lstm_vars)
    
    print('Dense variables:')
    dense_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=FLAGS.dense_scope)
    print_vars(dense_vars)
   
    train_lstm = tf.train.AdamOptimizer().minimize(loss_final, var_list  = lstm_vars+dense_vars)          
                                
 

    print('Ep variables:')
    Ep_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=FLAGS.Ep_scope)
    print_vars(Ep_vars)

    print('Ec variables:')
    Ec_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=FLAGS.Ec_scope)
    print_vars(Ec_vars)
    
    print('D variables:')
    D_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=FLAGS.D_scope)
    print_vars(D_vars)

      
    print('DEVICES')            
    get_available_gpus()
    print('DONE building')
        
    print('Trainable_vars')
    print_vars(tf.trainable_variables())
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    logdir = os.path.join(FLAGS.log_dir, FLAGS.run_name+'/')
    ckptdir = os.path.join(FLAGS.checkpoints_dir, FLAGS.run_name + '/')
    
    summary_op = tf.summary.merge_all() 
    summary_writer = tf.summary.FileWriter(logdir, sess.graph)
    
    init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    num_batches = 0
    
    
    Ep_loader = tf.train.Saver(Ep_vars)
    Ec_loader = tf.train.Saver(Ec_vars)
    D_loader = tf.train.Saver(D_vars)
    

        
    lstm_Saver = tf.train.Saver(lstm_vars, max_to_keep = 100)
    lstm_loader = tf.train.Saver(lstm_vars)
     
    
    if not os.path.exists(ckptdir):
        os.makedirs(ckptdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    

    try:
        i = 0    
        while not coord.should_stop(): 
        #for i in range(FLAGS.num_iters):
            print('step '+str(i))
            
            if(i==0):
                if(FLAGS.evaluate):
                    Ep_loader.restore(sess, FLAGS.restore_dir_Ep)
                    Ec_loader.restore(sess, FLAGS.restore_dir_Ec)
                    D_loader.restore(sess, FLAGS.restore_dir_D)
                    lstm_loader.restore(sess, FLAGS.restore_dir_lstm)

                else:
                    print('loading model for training...')
                    Ep_loader.restore(sess, FLAGS.restore_dir_Ep)
                    Ec_loader.restore(sess, FLAGS.restore_dir_Ec)
                    D_loader.restore(sess, FLAGS.restore_dir_D)
                    print('model loaded')
                  
            # write summary
            if(i%FLAGS.log_freq==0):
                print('Logging')
                summary = sess.run(summary_op)
                summary_writer.add_summary(summary,i)
                
            #Save model outputs also
            if(i%FLAGS.log_npy_freq==0):
                _, prediction_npy, vid_batch_npy, true_npy = sess.run([train_lstm, frame_predictions, video_batch,frame_true])
                np.save('tmp/npys_lstm/pred_'+str(i)+'_'+FLAGS.run_name+'.npy', prediction_npy)
                np.save('tmp/npys_lstm/orig_'+str(i)+'_'+FLAGS.run_name+'.npy', vid_batch_npy)
                np.save('tmp/npys_lstm/true_'+str(i)+'_'+FLAGS.run_name+'.npy', true_npy)
            else:
                sess.run([train_lstm])
           
            if(i%FLAGS.save_freq==0):
                lstm_Saver.save(sess, ckptdir+ 'lstm_and_dense/' ,global_step = i)
                           
            i+=1
    except Exception as e:
        coord.request_stop(e)

    finally:
        coord.request_stop()

    coord.join(threads)


def main():
    
    flags.DEFINE_integer('num_gpus', 1, 'number of gpus for training')
    flags.DEFINE_integer('batch_size', 20, 'size of batch_used')
    flags.DEFINE_integer('crop_height', 224, 'cropped height (randomnly cropped)')
    flags.DEFINE_integer('crop_width', 224, 'cropped width (randomnly cropped)')
    flags.DEFINE_integer('max_steps', 15, 'maximum frame difference (keep this smaller than batch_size')
    flags.DEFINE_integer('num_steps_total', 20, 'total number of steps lstm takes')
    flags.DEFINE_integer('num_steps_observed', 10, 'number of steps for which lstm observes input without supervision')
    flags.DEFINE_integer('max_start_index', 180, 'maximum frame index to get first frame for all videos')
    flags.DEFINE_integer('num_epochs_decoder', 1, 'number of epochs to run decoder training')
    flags.DEFINE_integer('num_epochs_pose_encoder', 1, 'num of epochs to train pose encoder')
    flags.DEFINE_integer('num_epochs_content_encoder', 1, 'num of epochs to train content encoder')
    flags.DEFINE_integer('resized_height',128,'Size to which resize image')
    flags.DEFINE_integer('resized_width', 128, 'Size to which resize image')
    flags.DEFINE_integer('num_epochs_discriminator', 1, 'num of epochs to train discriminator')
    flags.DEFINE_integer('num_epochs', 1000000, 'total number of training epochs')
    flags.DEFINE_integer('num_iters', 1000000, 'number of training iterations')
    flags.DEFINE_integer('size_pose_embedding', 5, 'size of pose embedding')
    flags.DEFINE_integer('size_content_embedding', 128, 'size of content embedding')
    flags.DEFINE_integer('state_size', 256, 'size of lstm state')
    flags.DEFINE_integer('num_layers', 2, 'num of lstm layers')
    flags.DEFINE_bool('load_all', False, 'whether to load everything or exclude lstm')
    flags.DEFINE_bool('training', True, 'whether training or not')
    flags.DEFINE_bool('evaluate', False, 'Whether generating video')
    flags.DEFINE_integer('log_freq', 10, 'freq to save summaries')
    flags.DEFINE_integer('log_npy_freq', 1000, 'frequency at which numpys are saved')
    flags.DEFINE_string('train_data_dir', '/mnt/AIDATA/datasets/kth/tfrecords_drnetsplit/train/', 'directory of training data(tfrecord files)')

    
    flags.DEFINE_string('dataset', 'kth', 'name of dataset, specified to reader')
    flags.DEFINE_string('log_dir', '/mnt/AIDATA/home/anmol/DrNet-tflow/logs_lstm/', 'Directory where to write logs')
    flags.DEFINE_string('run_name', '128x128_bs50_', 'name of run')
    flags.DEFINE_string('Ep_scope', 'Ep', 'scope of pose encoder')
    flags.DEFINE_string('Ec_scope', 'Ec', 'scope of content encoder')
    flags.DEFINE_string('D_scope', 'D', 'scope of decoder')
    flags.DEFINE_string('dense_scope', 'dense_after_lstm', 'scope of dense layer after lstm')
    flags.DEFINE_string('lstm_scope', 'multi_rnn_cell/', 'scope of lstm')

    flags.DEFINE_integer('save_freq', '500', 'how often to save model')          
    flags.DEFINE_string('restore_dir_Ep', '/mnt/AIDATA/home/anmol/DrNet-tflow/ckpt_6/test1/Ep/lol-60000', 'directory where to load Pose Encoder from')
    flags.DEFINE_string('restore_dir_Ec', '/mnt/AIDATA/home/anmol/DrNet-tflow/ckpt_6/test1/Ec/lol-60000', 'directory where to load Content Encoder from')

    flags.DEFINE_string('restore_dir_D', '/mnt/AIDATA/home/anmol/DrNet-tflow/ckpt_6/test1/D/lol-60000', 'directory where to load Decoder from')
    flags.DEFINE_string('restore_dir_lstm', '', 'directory to load lstm from')
    flags.DEFINE_string('checkpoints_dir', '/mnt/AIDATA/ckpt_lstm/', 'directory where checkpoints will be saved' )
    
    train()


if (__name__ == "__main__"):
    main()
