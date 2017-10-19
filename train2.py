import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import math 
import ipdb

from models.unet_vgg import Unet_models as Unet_model
from models.pose_encoder import pose_discriminator_tflow as C
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
    o = tf.cast(o, tf.float64)
    return -tf.reduce_mean(t*tf.log(o) + (1-t)*tf.log(1-o))


def print_vars(var_list):
    """Prints names of Tensors in var_list
    """
    names = [v.name for v in var_list]
    for name in names:
        print(name)

def train():
    """Trains DrNet Model

    """
    filenames = []
    bs_2 = int(FLAGS.batch_size/2)
    bs = FLAGS.batch_size 

    #Data reading pipeline
    for file in os.listdir(FLAGS.train_data_dir):
        if file.endswith('.tfrecords'):
            filenames.append(os.path.join(FLAGS.train_data_dir,file))
  
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(filenames, num_epochs = FLAGS.num_epochs)
    
    
    num_framestf=140
    
     


    k2=tf.random_uniform(shape = [],minval = 0, maxval=FLAGS.max_steps,dtype=tf.int64) #second frame is k2 ahead
    kp1 = tf.random_uniform(shape = [],minval = 0, maxval=FLAGS.max_steps,dtype=tf.int64) #pose encoder first index
    kc1 = tf.random_uniform(shape = [],minval = 0, maxval=FLAGS.max_steps,dtype=tf.int64) #scene discriminator 2
    kc2 = tf.random_uniform(shape = [],minval = 0, maxval=FLAGS.max_steps,dtype=tf.int64) #scene discriminator 2
    
    frame_list = [k2, kp1, kc1, kc2]
       
 
    video, label = reader.read_and_decode(filename_queue,bs,FLAGS.crop_height, FLAGS.crop_width,num_frames = 3, resized_height = FLAGS.resized_height, resized_width = FLAGS.resized_width, frame_is_random=True, rand_frame_list = FLAGS.max_steps, resize_image = True, crop_with_pad = True, rand_crop = False, resize_image_0=False, dataset = FLAGS.dataset, div_by_255 = True, max_steps = FLAGS.max_steps, train_drnet=True) #TODO 
    video_batch, label_batch = tf.train.shuffle_batch([video, label],
    batch_size=int(FLAGS.num_gpus*bs),
    num_threads=2,
    capacity=2000 + 3 * FLAGS.batch_size,
    min_after_dequeue=100)
    print('BATCH SHAPE')
    print(video_batch.get_shape())
    video_batch = tf.to_float(video_batch)
    tf.summary.scalar('video_mean', tf.reduce_mean(video_batch))
    
  
    model = Unet_model(FLAGS.size_pose_embedding, FLAGS.size_content_embedding)
    losses_C = []
    losses_Ep = []
    losses_Ec = []
    losses_D = []
    
    with tf.name_scope('drnet') as scope:    
        for i in range(FLAGS.num_gpus):
            reuse_flag = (i>0)
            with tf.name_scope('tower_'+str(i)), tf.device('/gpu:' + str(i)):
                print('GPU' + str(i))
                video_batch_tower = tf.slice(video_batch, begin = [i*bs, 0,0,0,0], size = [bs,-1,-1,-1,-1])
                
                tf.summary.scalar('INPUT_BATCH_TOWER', tf.reduce_mean(video_batch_tower))
                           
                
                #get all frames. 
                #images to train C
                images_1 = video_batch_tower[:,kc1,:,:,:] 
                images_4 = video_batch_tower[:,kc2,:,:,:] 
                
                #prepare batch to train C
                hp1 = model.Ep(images_1,FLAGS.Ep_scope, reuse_Ep = reuse_flag) #INIT E_P
                #hp1 = tf.concat([hp1, hp1],axis=0)
                tf.summary.scalar('hp1',tf.reduce_mean(hp1))
                tf.summary.histogram('hp1', hp1)                

                hp4 = model.Ep(images_4, FLAGS.Ep_scope, reuse_Ep = True)
                hp4 = tf.concat([hp4[:bs_2], tf.reverse(hp4[bs_2:],axis=[0])],axis=0)#keep first half from same scene, permute second half
                tf.summary.scalar('hp4', tf.reduce_mean(hp4))
                tf.summary.histogram('hp4', hp4)
                #C_batch = tf.squeeze(tf.concat([hp1, tf.stop_gradient(hp4)], axis = -1)) #stop gradient not really needed here nayway        
                C_batch = tf.squeeze(tf.concat([hp1, hp4], axis = -1)) 
                print(C_batch.shape) 
                #Get tower loss for C
                C_target_npy = np.ones(shape = [FLAGS.batch_size,1])
                C_target_npy[bs_2:,0] = 0
                C_target = tf.constant(C_target_npy, dtype=tf.float64)
                
                with tf.variable_scope(FLAGS.C_scope,reuse=reuse_flag) as C_scope:
                    C_out = C(C_batch) #make sure C_out doesn't backprop to Ep
                               
                tf.summary.scalar('Cout_duringtraining'+str(i), tf.reduce_mean(C_out))
                tf.summary.histogram('C_training_output'+str(i), C_out)
                #import ipdb
                #ipdb.set_trace()
                loss_C = bce(C_out, C_target)
                losses_C.append(loss_C)

                print('C variables:')
                C_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=FLAGS.C_scope)
                print_vars(C_vars)

                total_loss_C = FLAGS.beta*tf.reduce_mean(losses_C)
                tf.summary.scalar('total loss C', total_loss_C)
                train_C = tf.train.AdamOptimizer(learning_rate=FLAGS.lr).minimize(total_loss_C, var_list  = C_vars)
                with tf.control_dependencies([train_C]):
                    #images to train everything else
                    images_1_=video_batch_tower[:,0,:,:,:]
                    images_2 = video_batch_tower[:,k2,:,:,:] 
                    images_3 = video_batch_tower[:,kp1,:,:,:]
                   
                    
                    #Prepare batch to train Ep
                    hp3 = model.Ep(images_3, FLAGS.Ep_scope, reuse_Ep = True)
                    hp2 = model.Ep(images_2, FLAGS.Ep_scope, reuse_Ep = True)
                    tf.summary.scalar('hp2', tf.reduce_mean(hp2))                
                    tf.summary.histogram('hp2', hp2)
                    tf.summary.scalar('hp3', tf.reduce_mean(hp3))                
                    tf.summary.histogram('hp3', hp3)

                    C_batch_Ep = tf.squeeze(tf.concat([tf.stop_gradient(hp3), hp2], axis = -1)) 
                    
                     
                    #Get tower loss of E_p
                    C_target_Ep = tf.constant(0.5*np.ones(shape = [FLAGS.batch_size,1])) #maximize entropy
                    with tf.variable_scope(FLAGS.C_scope,reuse = True):
                        C_Ep_out = C(C_batch_Ep)
                    tf.summary.scalar('Cout_duringEptraining'+str(i), tf.reduce_mean(C_Ep_out))
                    tf.summary.histogram('Cout_during_Eptraining'+str(i), C_Ep_out)
                    loss_Ep = bce(C_Ep_out,C_target_Ep)
                    losses_Ep.append(loss_Ep)
                    
                   
                                                           
                    
                                
                    #Get tower loss of E_c
                    Ec_out = model.Ec(images_1_, FLAGS.Ec_scope, reuse_Ec = reuse_flag) #use only 1 frame
                    Ec_target = model.Ec(images_2,FLAGS.Ec_scope, reuse_Ec=True)
                   
                    tf.summary.scalar('Ec_target'+str(i), tf.reduce_mean(Ec_target))
                                                   
                    tf.summary.scalar('Ec_out'+str(i), tf.reduce_mean(Ec_out))
                                                   
                    #loss_Ec = tf.losses.mean_squared_error(Ec_out,tf.stop_gradient(Ec_target))
                    loss_Ec = tf.losses.mean_squared_error(tf.stop_gradient(Ec_target), Ec_out)
                    losses_Ec.append(loss_Ec)
                                    

                    #Get tower loss for D
                    out,hc1_ = model.D_hp_given(images_1_, FLAGS.Ec_scope, True, hp2, FLAGS.D_scope, reuse_D = reuse_flag, n_channels=1)
                    #import ipdb
                    #ipdb.set_trace()
                    loss_D = tf.losses.mean_squared_error(images_2[:,:,:,0], tf.squeeze(out))
                    losses_D.append(loss_D)

                 
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
                    

                    
                    total_loss_Ep = FLAGS.beta_2*tf.reduce_mean(losses_Ep)
                    tf.summary.scalar('total loss Ep', total_loss_Ep)

                    total_loss_Ec =FLAGS.alpha*tf.reduce_mean(losses_Ec)
                    tf.summary.scalar('total loss Ec', total_loss_Ec)

                    total_loss_D = FLAGS.alpha_2*tf.reduce_mean(losses_D)
                    tf.summary.scalar('total loss D', total_loss_D)

                    with tf.control_dependencies([total_loss_Ep, total_loss_Ec, total_loss_D]):
                        train_Ep = tf.train.AdamOptimizer(learning_rate=FLAGS.lr).minimize(total_loss_Ep, var_list = Ep_vars)
                        train_Ec = tf.train.AdamOptimizer(learning_rate=FLAGS.lr).minimize(total_loss_Ec, var_list = Ec_vars)
                        train_D = tf.train.AdamOptimizer(learning_rate=FLAGS.lr).minimize(total_loss_D, var_list = Ep_vars+Ec_vars+D_vars)
                        if(FLAGS.adv_loss):
                            train_ec_ep_d = tf.group(train_Ep, train_Ec, train_D)
                        else:
                            train_ec_ep_d = tf.group(train_Ec, train_D)
    
    dlEp_DEp = tf.gradients(total_loss_Ep, [hp2])
    dlD_DEp = tf.gradients(total_loss_D, [hp2])    
    dlC_DEp = tf.gradients(total_loss_C, [hp1])
    dlD_DEc = tf.gradients(total_loss_D, [hc1_])
    dlEc_DEc = tf.gradients(total_loss_Ec, [Ec_out])
    
    dd_ep1 = tf.gradients(total_loss_D, Ep_vars[-2])
    dep_ep1 = tf.gradients(total_loss_Ep, Ep_vars[-2])
    dc_ep1 = tf.gradients(total_loss_C, Ep_vars[-2])


    tf.summary.histogram('dlEp_DEp', dlEp_DEp)
    tf.summary.histogram('dlD_DEp', dlD_DEp)
    tf.summary.histogram('dlC_DEp', dlC_DEp) 
    tf.summary.histogram('dlD_DEc', dlD_DEc)
    tf.summary.histogram('dlEc_DEc', dlEc_DEc)

    tf.summary.histogram('dd_ep1', dd_ep1)
    tf.summary.histogram('dep_ep1', dep_ep1)
    tf.summary.histogram('dc_ep1', dc_ep1)
 
    
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
    
    C_Saver = tf.train.Saver(C_vars, max_to_keep = 200, write_version=tf.train.SaverDef.V2)
    Ep_Saver = tf.train.Saver(Ep_vars, max_to_keep = 200, write_version=tf.train.SaverDef.V2)
    Ec_Saver = tf.train.Saver(Ec_vars, max_to_keep = 200, write_version=tf.train.SaverDef.V2)
    D_Saver = tf.train.Saver(D_vars, max_to_keep = 200, write_version=tf.train.SaverDef.V2)
    
    if not os.path.exists(ckptdir):
        os.makedirs(ckptdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)


    try:
        i = 0    
        while not coord.should_stop(): 
            print('step '+str(i))
            #import ipdb;    
            #ipdb.set_trace()
            
            # write summary
            if(i%FLAGS.log_freq==0):
                print('Logging')
                summary = sess.run(summary_op)
                summary_writer.add_summary(summary,i)
                
            #Save model outputs also
            if(i%FLAGS.check_output_freq==0):    
                #ipdb.set_trace()
                _, _,orig1, orig2, pred2 = sess.run([train_C, train_ec_ep_d, images_1_, images_2,out])
                
                np.save('tmp/npys/orig1_'+str(i)+'_'+FLAGS.run_name+'_.npy', orig1)
                np.save('tmp/npys/orig2_'+str(i)+'_'+FLAGS.run_name+'_.npy', orig2)
                np.save('tmp/npys/pred2_'+str(i)+'_'+FLAGS.run_name+'_.npy', pred2)
                #np.save('tmp/npys/orig2_shuffled_'+str(i)+'_'+FLAGS.run_name+'_.npy',orig2_shuffled)
            
            else:
                 sess.run([train_C, train_ec_ep_d])
                
            if(i%FLAGS.save_freq==0):
                C_Saver.save(sess, ckptdir+ 'C/lol' ,global_step = i, write_meta_graph = False)
                Ep_Saver.save(sess, ckptdir + 'Ep/lol', global_step = i, write_meta_graph = False)
                Ec_Saver.save(sess, ckptdir+  'Ec/lol', global_step  = i, write_meta_graph = False)
                D_Saver.save(sess, ckptdir + 'D/lol', global_step = i, write_meta_graph = False)
            
            i+=1
    except Exception as e:
        coord.request_stop(e)

    finally:
        coord.request_stop()

    coord.join(threads)


def main():
    
    flags.DEFINE_integer('num_gpus', 1, 'number of gpus for training')
    flags.DEFINE_integer('batch_size', 20, 'size of batch_used')
    flags.DEFINE_integer('crop_height', None, 'cropped height (randomnly cropped)')
    flags.DEFINE_integer('crop_width', None, 'cropped width (randomnly cropped)')
    flags.DEFINE_integer('max_steps', 15, 'maximum frame difference (keep this smaller than batch_size')
    flags.DEFINE_integer('num_frames_sampled', 200, 'number of frames to extract from video (evenly sampled)')
    flags.DEFINE_integer('num_epochs_decoder', 1, 'number of epochs to run decoder training')
    flags.DEFINE_integer('num_epochs_pose_encoder', 1, 'num of epochs to train pose encoder')
    flags.DEFINE_integer('num_epochs_content_encoder', 1, 'num of epochs to train content encoder')
    flags.DEFINE_integer('resized_height',128,'Size to which resize image')
    flags.DEFINE_integer('resized_width', 128, 'Size to which resize image')
    flags.DEFINE_integer('num_epochs_discriminator', 1, 'num of epochs to train discriminator')
    flags.DEFINE_integer('num_epochs', 10, 'total number of training epochs')
    flags.DEFINE_integer('num_iters', 100000, 'number of training iterations')
    flags.DEFINE_integer('size_pose_embedding', 5, 'size of pose embedding')
    flags.DEFINE_integer('size_content_embedding', 128, 'size of content embedding')
    flags.DEFINE_integer('log_freq', 10, 'freq to save summaries')
    flags.DEFINE_integer('check_output_freq', 100, 'how often to check output (to make sure things are working)')
    flags.DEFINE_float('alpha', 1.0, 'alpha term in total loss')
    flags.DEFINE_float('alpha_2', 1.0, 'alpha for decoder loss')
    flags.DEFINE_float('beta', 0.001, 'beta term in total loss')
    flags.DEFINE_float('beta_2', 0.001, 'another thing')
    flags.DEFINE_float('lr', '0.002', 'learning rate')
    flags.DEFINE_string('train_data_dir', '/mnt/AIDATA/datasets/kth/tfrecords_drnetsplit/train/', 'directory of training data(tfrecord files)')
    flags.DEFINE_string('dataset', 'kth', 'name of dataset, specified to reader')
    flags.DEFINE_string('log_dir', '/mnt/AIDATA/home/anmol/DrNet-tflow/logs_6/', 'Directory where to write logs')
    flags.DEFINE_string('run_name', 'init', 'name of run')
    flags.DEFINE_string('C_scope', 'C', 'variable scope of discriminator')
    flags.DEFINE_string('Ep_scope', 'Ep', 'scope of pose encoder')
    flags.DEFINE_string('Ec_scope', 'Ec', 'scope of content encoder')
    flags.DEFINE_string('D_scope', 'D', 'scope of decoder')
    flags.DEFINE_integer('save_freq', '200', 'how often to save model')          
    flags.DEFINE_string('checkpoints_dir', '/mnt/AIDATA/home/anmol/DrNet-tflow/ckpt_5', 'directory where checkpoints will be saved' )
    
    flags.DEFINE_bool('train_encoders', False, 'whether to train encoders')
    flags.DEFINE_bool('train_prediction', True, 'whether to train video prediction model')
    flags.DEFINE_bool('adv_loss', True, 'whether to do adversarial training')
    train()


if (__name__ == "__main__"):
    main()
