import tensorflow as tf
from proposed_network import proposed_network
import numpy as np
from ops import losses
from ops import utils
from glob import glob
import cv2
import time
from Unet import Unet
import random
import re
from PIL import Image
import os
from tensorflow.contrib.tensorboard.plugins import projector


def train_with_gpu(flag):
    with tf.Graph().as_default():
#read data
        holo = glob('/path/to/data/*.bmp')
        intensity = glob('/path/to/data/*.bmp')
        phase = glob('/path/to/data/*.bmp')

        num_samples_per_epoch = len(holo)
        print ("num_samples_per_epoch: %d "%num_samples_per_epoch)
        
#setting optimization parameter
        num_batches_per_epoch = num_samples_per_epoch // flag.batch_size

        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0), trainable=False)

        decay_steps = int(num_batches_per_epoch * flag.num_epochs_per_decay)

        lr = tf.train.exponential_decay(flag.initial_learning_rates,
                                        global_step,
                                        decay_steps,
                                        flag.learning_rate_decay_factor,
                                        staircase=True)
        
                                        
#empty tensor for input and output
        input_placeholder = tf.placeholder(tf.float32,
                                           [flag.batch_size, flag.image_height, flag.image_width, flag.channel_dim])
        output_intensity_placeholder = tf.placeholder(tf.float32,
                                           [flag.batch_size, flag.image_height, flag.image_width, flag.channel_dim])
        output_phase_placeholder = tf.placeholder(tf.float32,
                                           [flag.batch_size, flag.image_height, flag.image_width, flag.channel_dim])
        
        
        model = proposed_network()
#resize the image 
        input_ = tf.image.resize_images(input_placeholder[:, :, :, :], [flag.image_height//2, flag.image_width//2])
        target_intensity =  tf.image.resize_images(output_intensity_placeholder[:, :, :, :],[flag.image_height//2, flag.image_width//2])
        target_phase =  tf.image.resize_images(output_phase_placeholder[:, :, :, :],[flag.image_height//2, flag.image_width//2])
        target_phase_0_1 = model.Normalize_0_1(target_phase) 
         
#intensity & phase input as two channel of a image
        target_0_255 = tf.concat([target_intensity, 255*target_phase_0_1], 3)
        target_0_1 = model.Normalize_0_1(target_0_255)
    
        with tf.name_scope('summaries'):
          with tf.name_scope('Model'):
            with tf.device('/gpu:1'):
                encode = model.inference_encode(input_)
            with tf.device('/gpu:0'):
                output_0_255, output_0_1 = model.inference_decode(encode)
            

            output_intensity, output_phase = tf.split(output_0_1, [1,1], 3) 
            
#with regulization: 
          loss = tf.losses.mean_squared_error(target_0_1, output_0_1) + flag.coef * tf.reduce_mean(model.total_variation(output_0_1))
              
          tf.summary.scalar('loss', loss)
          tf.summary.scalar('loss_phase', loss_phase)
        

        var_list = tf.trainable_variables()

        optimizer = tf.train.AdamOptimizer(lr)
        grads_and_vars = optimizer.compute_gradients(loss, var_list=var_list)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)

        init = tf.initialize_all_variables()
       
#setting for GPU
        config = tf.ConfigProto()
        config.allow_soft_placement=True
        config.gpu_options.allow_growth = True
        
#other possible setting
        #config.log_device_placement = True
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        
#for visulization using tensorboard
        merged_summary_op = tf.summary.merge_all()

#start to run the session
        with tf.Session(config=config) as sess:
          
          for train in flag.train_phase:
           print ("------------------------------------")
           print ("train phase is now: %d"%train)
         
           if (train == 1):
               model = FusionNet()
               print ('Training model: proposed_network with noise level %d'%flag.noise_total)
          
           for noise_level in flag.noise_total:
            #for learning_rate in flag.initial_learning_rates:
            sess.run(init)
          
#save log file in certain path
            if  train == 1:
                log = './train_proposed_network_log'
                if (tf.gfile.Exists(log)== False):
                    print ('The specified base directory does not exist, now is making directory!')
                    os.mkdir(log_1)
                    print ('The specified directory is created successfully!')
                    
                summary_writer = tf.summary.FileWriter(log_1, graph_def=sess.graph_def)
                projector.visualize_embeddings(summary_writer, projector.ProjectorConfig())
                ckpt_dir = "./checkpoint_without_noise_FusionNet_batch1_newImage"
            print("Learning start!!")
            
            start = time.time()

            ckpt_cnt = 1
            #if utils.load_ckpt(flag.ckpt_dir, sess, flag.ckpt_name):
            if utils.load_ckpt(ckpt_dir, sess, flag.ckpt_name):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
            
            epoch = 0
            
            
#Get validation index

            #val_index_input, val_index_output_intensity, val_index_output_phase, val_sample = model.Val_GetIndexList_newImage("/home/zmxu/dataset/single_depth_10layers_256x256/holo", "/home/zmxu/dataset/single_depth_10layers_256x256/ground_truth" ,"/home/zmxu/dataset/single_depth_10layers_256x256/ground_truth")
            
            print ("Noise level is: %d"%noise_level)
            
            while (epoch < flag.total_epoch ):
                sample = {}
                num_batches = num_batches_per_epoch
            
#get index of train image
                for batch_idx in range(num_batches):
                    if (sample == {}): 
                        whole_new_range = range(0, flag.range_)
                        #new_range = [i for i in whole_new_range if i not in val_sample]
                        new_range = whole_new_range
                    else:
                        new_range = [i for i in new_range if i not in sample]
                        
                    index_input, index_output_intensity, index_output_phase, sample = model.GetIndexList("/path/to/input/data", "/path/to/groundtruth/intensity" ,"/path/to/groundtruth/phase",  len(flag.data_in_name), flag.batch_size, batch_idx, new_range)
#read image and add noise
                    noise = np.zeros((flag.image_height, flag.image_width, 1), np.int8)
                    noises = cv2.randn(noise, np.zeros(1), np.ones(1)*noise_level)
                    
                    batch_in_files = [holo[idx] for idx in index_input]
                    batch_in_image = [(cv2.imread(batch_in_file)+ noises) for batch_in_file in batch_in_files]
                  
                    if flag.channel_dim == 1:
                        batch_in_images = np.array(batch_in_image).astype(np.float32)[:, :, :, :1]
                    else:
                        batch_in_images = np.array(batch_in_image).astype(np.float32)
                        
                    batch_out_intesnity_files = [intensity[idx] for idx in index_output_intensity]
                    batch_out_intensity_image = [cv2.imread(batch_out_intesnity_file) for batch_out_intesnity_file in batch_out_intesnity_files]
                    if flag.channel_dim == 1:
                        batch_out_intensity_images = np.array(batch_out_intensity_image).astype(np.float32)[:, :, :, :1]
                    else:
                        batch_out_intensity_images = np.array(batch_out_intensity_image).astype(np.float32)
                        
                    batch_out_phase_files = [phase[idx] for idx in index_output_phase]
                    batch_out_phase_image = [cv2.imread(batch_out_phase_file) for batch_out_phase_file in batch_out_phase_files]
                    if flag.channel_dim == 1:
                        batch_out_phase_images = np.array(batch_out_phase_image).astype(np.float32)[:, :, :, :1]
                    else:
                        batch_out_phase_images = np.array(batch_out_phase_image).astype(np.float32)

#feed the value to the empty tensor
                    feed = {input_placeholder: batch_in_images, output_intensity_placeholder: batch_out_intensity_images, output_phase_placeholder: batch_out_phase_images}
                 

                    sess.run(train_op, feed_dict=feed)
                    
#stop using regulization when current epoch >=loss_function_change_epoch to prevent overlearning
                  
                    print("(%ds) Epoch: %d[%d/%d], loss: %f" % (time.time() - start, epoch, batch_idx,
                                                               num_batches_per_epoch-1, sess.run(loss, feed)))
                       
#run and write the log file(for tensorboard visulization)                                        
                    summaries = sess.run(merged_summary_op, feed)
                    summary_writer.add_summary(summaries, epoch*num_batches_per_epoch + batch_idx)

#save the model every 400 batch
                    ckpt_cnt += 1         
                    if np.mod(ckpt_cnt, 400) == 2:
                        #utils.save_ckpt(flag.ckpt_dir, ckpt_cnt, sess, flag.ckpt_name)
                        utils.save_ckpt(ckpt_dir, ckpt_cnt, sess, flag.ckpt_name)
                        j = 0
                       
                               
                        if train == 1:
                            dir = './train_image'
                            if (tf.gfile.Exists(dir)== False):
                                print ('The specified base directory does not exist, now is making directory!')
                                os.mkdir(dir)
                                print ('The specified directory is created successfully!')
                            for idx in range(len(sample)): 
                               cv2.imwrite(dir +'/coef%d_gen_intensity_with_noise_%d_%s.bmp'%(flag.coef, noise_level, sample[idx]), 255*output_intensity[j].eval(feed_dict=feed))
                               cv2.imwrite(dir +'/coef%d_gen_phase_with_noise_%d_%s.bmp'%(flag.coef, noise_level, sample[idx]), 10*output_phase[j].eval(feed_dict=feed))
                               j += 1
                     
                epoch += 1
            
                
def train_with_cpu(flag):
    pass
