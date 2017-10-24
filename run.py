''' Script to train DrNet. Please see README.md for details

'''


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import math 
import ipdb


#Import the models here
#from models.unet_vgg import Unet_models as Unet_model
#from models.unet_dcgan64 import Unet_models as Unet_model
from models.unet_dcgan import Unet_models as Unet_model
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
    
    
    #Frame indices used for training different parts of the graph 
    k2 = 1
    kp1 = 2
    kc1 = 3
    kc2 = 4
    
      
    with tf.device('/cpu:0'):
        video, label = reader.read_and_decode(filename_queue,bs,FLAGS.crop_height, FLAGS.crop_width,num_frames = 3, resized_height = FLAGS.resized_height, resized_width = FLAGS.resized_width, frame_is_random=True, rand_frame_list = None, resize_image = True, crop_with_pad = False, rand_crop = False, resize_image_0=False, dataset = FLAGS.dataset, div_by_255 = True, max_steps = FLAGS.max_steps, train_drnet=True) 
        video_batch, label_batch = tf.train.shuffle_batch([video, label],
        batch_size=int(FLAGS.num_gpus*FLAGS.batch_size),
        num_threads=12,
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
    
    #First define graph for training scene discriminator 
    with tf.name_scope('drnet') as scope:    
        C_target_npy = np.ones(shape = [FLAGS.batch_size,1])
        C_target_npy[bs_2:,0] = 0
        C_target = tf.constant(C_target_npy, dtype=tf.float64)
        for i in range(FLAGS.num_gpus):
            reuse_flag = (i>0)
            with tf.name_scope('tower_Closs_'+str(i)):#, tf.device('/gpu:' + str(i)):
                print('GPU' + str(i))
                video_batch_tower = tf.slice(video_batch, begin = [i*bs, 0,0,0,0], size = [bs,-1,-1,-1,-1])
                
                tf.summary.scalar('INPUT_BATCH_TOWER', tf.reduce_mean(video_batch_tower))
                           
                
                #get all frames. 
                #images to train C
                images_1 = video_batch_tower[:,kc1,:,:,:] 
                images_4 = video_batch_tower[:,kc2,:,:,:] 
                
                #prepare batch to train C
                hp1 = model.Ep(images_1,FLAGS.Ep_scope, reuse_Ep = reuse_flag) #INIT E_P
                tf.summary.scalar('hp1',tf.reduce_mean(hp1))
                tf.summary.histogram('hp1', hp1)                

                hp4 = model.Ep(images_4, FLAGS.Ep_scope, reuse_Ep = True)
                hp4 = tf.concat([hp4[:bs_2], tf.reverse(hp4[bs_2:],axis=[0])],axis=0)#keep first half from same scene, permute second half
                tf.summary.scalar('hp4', tf.reduce_mean(hp4))
                tf.summary.histogram('hp4', hp4)
                       
                C_batch = tf.squeeze(tf.concat([hp1, hp4], axis = -1)) 
                print(C_batch.shape) 
                
                #Get tower loss for C
                with tf.variable_scope(FLAGS.C_scope,reuse=reuse_flag) as C_scope:
                    C_out = C(C_batch) #make sure C_out doesn't backprop to Ep
                               
                tf.summary.scalar('Cout_duringtraining'+str(i), tf.reduce_mean(C_out))
                tf.summary.histogram('C_training_output'+str(i), C_out)
                
                loss_C = bce(C_out, C_target)
                losses_C.append(loss_C)

                print('C variables:')
                C_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=FLAGS.C_scope)
                print_vars(C_vars)

        total_loss_C = FLAGS.beta*tf.reduce_mean(losses_C)
        tf.summary.scalar('total loss C', total_loss_C)
        train_C = tf.train.AdamOptimizer(learning_rate=FLAGS.lr).minimize(total_loss_C, var_list  = C_vars)
        
        C_target_Ep = tf.constant(0.5*np.ones(shape = [FLAGS.batch_size,1]), dtype=tf.float64) #maximize entropy        
        
        for i in range(FLAGS.num_gpus):
            with tf.name_scope('tower_D_Ep_Ec_loss_'+str(i)), tf.control_dependencies([train_C]):#, tf.device('/gpu:' + str(i)), 
                reuse_flag = (i>0)
                print('GPU' + str(i))
                video_batch_tower = tf.slice(video_batch, begin = [i*FLAGS.batch_size, 0,0,0,0], size = [FLAGS.batch_size,-1,-1,-1,-1])
                 
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
        

        
        total_loss_Ep = tf.cast(FLAGS.beta_2*tf.reduce_mean(losses_Ep), tf.float64)
        tf.summary.scalar('total loss Ep', total_loss_Ep)

        total_loss_Ec = tf.cast(FLAGS.alpha*tf.reduce_mean(losses_Ec), tf.float64)
        tf.summary.scalar('total loss Ec', total_loss_Ec)

        total_loss_D = tf.cast(FLAGS.alpha_2*tf.reduce_mean(losses_D), tf.float64)
        tf.summary.scalar('total loss D', total_loss_D)
        
        with tf.control_dependencies([total_loss_Ep, total_loss_Ec, total_loss_D]):
            if(FLAGS.adv_loss):
                total_loss = total_loss_Ep + total_loss_Ec + total_loss_D
                var_list_total_loss = Ep_vars + Ec_vars + D_vars
            else:
                total_loss = total_loss_Ec + total_loss_D
                var_list_total_loss = Ec_vars + D_vars
            train_ec_ep_d = tf.train.AdamOptimizer(learning_rate=FLAGS.lr, beta1=0.5).minimize(total_loss, var_list=var_list_total_loss)
    
    
    config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
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
    C_loader = tf.train.Saver(C_vars)
    
    Ep_Saver = tf.train.Saver(Ep_vars, max_to_keep = 200, write_version=tf.train.SaverDef.V2)
    Ep_loader = tf.train.Saver(Ep_vars)
    
    Ec_Saver = tf.train.Saver(Ec_vars, max_to_keep = 200, write_version=tf.train.SaverDef.V2)
    Ec_loader = tf.train.Saver(Ec_vars)
    
    D_Saver = tf.train.Saver(D_vars, max_to_keep = 200, write_version=tf.train.SaverDef.V2)
    D_loader = tf.train.Saver(D_vars)


    if not os.path.exists(ckptdir):
        os.makedirs(ckptdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)


    try:
        i = 0
        if(FLAGS.load_ckpt):
            print('loading checkpoint...')
            C_loader.restore(sess, FLAGS.restore_dir_C)
            Ep_loader.restore(sess, FLAGS.restore_dir_Ep)
            Ec_loader.restore(sess, FLAGS.restore_dir_Ec)
            D_loader.restore(sess, FLAGS.restore_dir_D)
            print('checkpoint loaded')
        while not coord.should_stop(): 
            print('step '+str(i))
                        
            # write summary
            if(i%FLAGS.log_freq==0):
                print('Logging')
                summary = sess.run(summary_op)
                summary_writer.add_summary(summary,i)

                
            #Save model outputs also
            if(i%FLAGS.check_output_freq==0):    
                _, _,orig1, orig2, pred2 = sess.run([train_C, train_ec_ep_d, images_1_, images_2,out])
                
                np.save('/mnt/AIDATA/home/anmol/DrNet-tflow/tmp/npys/orig1_'+str(i)+'_'+FLAGS.run_name+'_.npy', orig1)
                np.save('/mnt/AIDATA/home/anmol/DrNet-tflow/tmp/npys/orig2_'+str(i)+'_'+FLAGS.run_name+'_.npy', orig2)
                np.save('/mnt/AIDATA/home/anmol/DrNet-tflow/tmp/npys/pred2_'+str(i)+'_'+FLAGS.run_name+'_.npy', pred2)
            else:
                sess.run([train_C, train_ec_ep_d])
                
            if(i%FLAGS.save_freq==0):
                C_Saver.save(sess, ckptdir+ 'C/ckpt' ,global_step = i, write_meta_graph = False)
                Ep_Saver.save(sess, ckptdir + 'Ep/ckpt', global_step = i, write_meta_graph = False)
                Ec_Saver.save(sess, ckptdir+  'Ec/ckpt', global_step  = i, write_meta_graph = False)
                D_Saver.save(sess, ckptdir + 'D/ckpt', global_step = i, write_meta_graph = False)
            
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
    flags.DEFINE_integer('check_output_freq', 500, 'how often to check output (to make sure things are working)')
    flags.DEFINE_float('alpha', 1.0, 'alpha term in total loss')
    flags.DEFINE_float('alpha_2', 1.0, 'alpha for decoder loss')
    flags.DEFINE_float('beta', 0.0001, 'beta term in total loss')
    flags.DEFINE_float('beta_2', 0.0001, 'another thing')
    flags.DEFINE_float('lr', '0.002', 'learning rate')
    flags.DEFINE_string('train_data_dir', '/mnt/AIDATA/datasets/kth/tfrecords_drnetsplit/train/', 'directory of training data(tfrecord files)')
    flags.DEFINE_string('dataset', 'kth', 'name of dataset, specified to reader')
    flags.DEFINE_string('log_dir', '/mnt/AIDATA/home/anmol/DrNet-tflow/logs_8/', 'Directory where to write logs')
    flags.DEFINE_string('run_name', 'init', 'name of run')
    flags.DEFINE_string('C_scope', 'C', 'variable scope of discriminator')
    flags.DEFINE_string('Ep_scope', 'Ep', 'scope of pose encoder')
    flags.DEFINE_string('Ec_scope', 'Ec', 'scope of content encoder')
    flags.DEFINE_string('D_scope', 'D', 'scope of decoder')
    flags.DEFINE_integer('save_freq', '1000', 'how often to save model')          
    flags.DEFINE_string('checkpoints_dir', '/mnt/AIDATA/home/anmol/DrNet-tflow/ckpt_6', 'directory where checkpoints will be saved' )
    flags.DEFINE_string('restore_dir_C', '/mnt/AIDATA/home/anmol/DrNet-tflow/ckpt_5/dcganUnet_dcganep_bs50_128x128_threads12/C/lol-20000', 'directory to load scene discriminator checkpoint')
    flags.DEFINE_string('restore_dir_Ep', '/mnt/AIDATA/home/anmol/DrNet-tflow/ckpt_5/dcganUnet_dcganep_bs50_128x128_threads12/Ep/lol-20000', 'directory to load pose encoder checkpoint')
    flags.DEFINE_string('restore_dir_Ec', '/mnt/AIDATA/home/anmol/DrNet-tflow/ckpt_5/dcganUnet_dcganep_bs50_128x128_threads12/Ec/lol-20000', 'directory to load content encoder checkpoint')
    flags.DEFINE_string('restore_dir_D', '/mnt/AIDATA/home/anmol/DrNet-tflow/ckpt_5/dcganUnet_dcganep_bs50_128x128_threads12/D/lol-20000', 'directory to load decoder checkpoint')

    flags.DEFINE_bool('load_ckpt', False, 'whether to restore a checkpoint before training')
    flags.DEFINE_bool('adv_loss', True, 'whether to do adversarial training')
    train()


if (__name__ == "__main__"):
    main()
