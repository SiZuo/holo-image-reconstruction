import tensorflow as tf
from FusionNet_newImage import FusionNet
#from FusionNet_add_1 import FusionNet_add_1
#from FusionNet_del_1 import FusionNet_del_1
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


def train_with_cpu(flag):
    with tf.Graph().as_default():
#read data
        holo = glob('/home/zmxu/dataset/obj/holo/*.bmp')
        intensity = glob('/home/zmxu/dataset/obj/gt_intensity/*.bmp')
        phase = glob('/home/zmxu/dataset/obj/gt_phase/*.bmp')

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
        
        
        model = FusionNet()
                                                                       #resize the image 
        input_ = tf.image.resize_images(input_placeholder[:, :, :, :], [flag.image_height//2, flag.image_width//2])
        target_intensity =  tf.image.resize_images(output_intensity_placeholder[:, :, :, :],[flag.image_height//2, flag.image_width//2])
        target_phase =  tf.image.resize_images(output_phase_placeholder[:, :, :, :],[flag.image_height//2, flag.image_width//2])
        target_phase_0_1 = model.Normalize_0_1(target_phase) 
         
#intensity & phase input as two channel of a image
        target_0_255 = tf.concat([target_intensity, 255*target_phase_0_1], 3)
        target_0_1 = model.Normalize_0_1(target_0_255)
    
        '''
        #target_original = tf.concat([target_intensity, target_phase], 3)
        #target_0_225 = tf.concat([target_intensity, target_phase], 3)
        #target_0_1 = tf.concat([target_intensity_0_1, target_phase_0_1], 3)
        #print("target_0_225", target_0_225.get_shape())
       
        '''
        with tf.name_scope('summaries'):
          with tf.name_scope('Model'):
            with tf.device('/gpu:1'):
                encode = model.inference_encode(input_)
            with tf.device('/gpu:0'):
                output_0_255, output_0_1 = model.inference_decode(encode)
            

            output_intensity, output_phase = tf.split(output_0_1, [1,1], 3) 
          
          #loss = tf.losses.mean_squared_error(target_0_1, output_0_1) 
          loss_phase =  tf.losses.mean_squared_error(target_phase_0_1, output_phase)
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
           '''
           if (train == 2):
               model = FusionNet_del_1()
               print ('Training model: FusionNet_del_1_layer with noise level 15')
           if (train == 0):
               model = FusionNet_add_1()
               print ('Training model: FusionNet_add_1_layer with noise level 15')
            '''
           if (train == 1):
               model = FusionNet()
               print ('Training model: FusionNet with noise level %d with BN at the begining'%flag.noise_total)
          
           for noise_level in flag.noise_total:
            #for learning_rate in flag.initial_learning_rates:
            sess.run(init)
            '''
            
            #if learning_rate == 0.003:
             #   summary_writer = tf.summary.FileWriter('./train_noise_5_gradient_logs_2_003', graph_def=sess.graph_def)
             #   ckpt_dir = "./checkpoint_with_noise_5_gradient_2_003"
            #if learning_rate == 0.004:
             #   summary_writer = tf.summary.FileWriter('./train_noise_5_gradient_logs_2_004', graph_def=sess.graph_def)
              #  ckpt_dir = "./checkpoint_with_noise_5_gradient_2_004"
            #if learning_rate == 0.0009:
             #   summary_writer = tf.summary.FileWriter('./train_noise_5_gradient_logs_2_0009', graph_def=sess.graph_def)
              #  ckpt_dir = "./checkpoint_with_noise_5_gradient_2_0009"
            #if learning_rate == 0.0007:
             #   summary_writer = tf.summary.FileWriter('./train_noise_5_gradient_logs_2_0007', graph_def=sess.graph_def)
              #  ckpt_dir = "./checkpoint_with_noise_5_gradient_2_0007"
          #  if  train == 0:
           #     log = './train_noise_15_log_FusionNet_add_1_layer'
                #os.mkdir(log)
            #    summary_writer = tf.summary.FileWriter(log, graph_def=sess.graph_def)
              #  projector.visualize_embeddings(summary_writer, projector.ProjectorConfig())
             #   ckpt_dir = "./checkpoint_with_noise_15_FusionNet_add_1_layer"
            '''
#save log file in certain path
            if  train == 1:
                log_1 = './train_noise_0_log_FusionNet_batch1_BNatBegin_newImage'
                if (tf.gfile.Exists(log_1)== False):
                    print ('The specified base directory does not exist, now is making directory!')
                    os.mkdir(log_1)
                    print ('The specified directory is created successfully!')
                    
                summary_writer = tf.summary.FileWriter(log_1, graph_def=sess.graph_def)
                projector.visualize_embeddings(summary_writer, projector.ProjectorConfig())
                ckpt_dir = "./checkpoint_without_noise_FusionNet_batch1_BNatBegin_new_newImage"
            '''
            if  train == 2:
                log_2 = './train_noise_15_log_FusionNet_del_1_layer_batch1_newImage'
                #os.mkdir(log_2)
                summary_writer = tf.summary.FileWriter(log_2, graph_def=sess.graph_def)
                projector.visualize_embeddings(summary_writer, projector.ProjectorConfig())
                ckpt_dir = "./checkpoint_with_noise_15_FusionNet_del_1_layer_batch10"
                
            #if  flag.coef == 3:
             #   summary_writer = tf.summary.FileWriter('./train_noise_15_logs_gradient_3_7epo', graph_def=sess.graph_def)
              #  projector.visualize_embeddings(summary_writer, projector.ProjectorConfig())
               # ckpt_dir = "./checkpoint_with_noise_15_gradient_3_7epo"
            
            #if  flag.coef == 0.5:
             #   summary_writer = tf.summary.FileWriter('./train_noise_15_logs_gradient_0_5_7epo', graph_def=sess.graph_def)
              #  projector.visualize_embeddings(summary_writer, projector.ProjectorConfig())
               # ckpt_dir = "./checkpoint_with_noise_15_gradient_0_5_7epo"
                
            #if  flag.coef == 2.5:
             #   summary_writer = tf.summary.FileWriter('./train_noise_15_logs_gradient_2_5', graph_def=sess.graph_def)
              #  projector.visualize_embeddings(summary_writer, projector.ProjectorConfig())
               # ckpt_dir = "./checkpoint_with_noise_15_gradient_2_5"
            
           # if  flag.coef == 8:
            #    summary_writer = tf.summary.FileWriter('./train_noise_15_logs_gradient_8', graph_def=sess.graph_def)
             #   projector.visualize_embeddings(summary_writer, projector.ProjectorConfig())
              #  ckpt_dir = "./checkpoint_with_noise_15_gradient_8"
                
            #if  train == 0:
             #   summary_writer = tf.summary.FileWriter('./train_noise_15_logs_gradient_0_5', graph_def=sess.graph_def)
              #  projector.visualize_embeddings(summary_writer, projector.ProjectorConfig())
               # ckpt_dir = "./checkpoint_with_noise_15_gradient_0_5"
                
            #if  train == 1:
             #   summary_writer = tf.summary.FileWriter('./train_noise_15_logs_gradient_0_5_3epo', graph_def=sess.graph_def)
              #  projector.visualize_embeddings(summary_writer, projector.ProjectorConfig())
               # ckpt_dir = "./checkpoint_with_noise_15_gradient_0_5_3epo"
            '''
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
                        
                    index_input, index_output_intensity, index_output_phase, sample = model.GetIndexList("/home/zmxu/dataset/obj/holo", "/home/zmxu/dataset/obj/gt_intensity" ,"/home/zmxu/dataset/obj/gt_phase",  len(flag.data_in_name), flag.batch_size, batch_idx, new_range)
#read image and add noise
                    noise = np.zeros((flag.image_height, flag.image_width, 1), np.int8)
                    noises = cv2.randn(noise, np.zeros(1), np.ones(1)*noise_level)
                    
                    batch_in_files = [holo[idx] for idx in index_input]
                    batch_in_image = [(cv2.imread(batch_in_file)+ noises) for batch_in_file in batch_in_files]
                    
#fix the pixel value in the range(0, 255)
                    #batch_in_image[batch_in_image < 0] = 0
                    #batch_in_image[batch_in_image > 255] = 255
                    #batch_in_image[batch_in_image < 0] = 0
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
                    '''
                    if(epoch >= loss_function_change_epoch ):
                        flag.coef = 0
                        print("(%ds) Epoch: %d[%d/%d], loss: %f" % (time.time() - start, epoch, batch_idx, 
                                                               num_batches_per_epoch-1, sess.run(loss, feed)))
                    else:
                        print("(%ds) Epoch: %d[%d/%d], loss: %f" % (time.time() - start, epoch, batch_idx,
                                                               num_batches_per_epoch-1, sess.run(loss, feed)))
                   '''
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
                        '''
                        if learning_rate == 0.003:
                            for idx in range(len(sample)): 
                               cv2.imwrite('./train_noise_5_gradient_result_2_003/gen_intensity_with_noise_%s.bmp'%(sample[idx]), 255*output_intensity[j].eval(feed_dict=feed))
                               cv2.imwrite('./train_noise_5_gradient_result_2_003/gen_phase_with_noise_%s.bmp'%(sample[idx]), 10*output_phase[j].eval(feed_dict=feed))
                               j += 1
                            
                        if learning_rate == 0.004:
                            for idx in range(len(sample)): 
                               cv2.imwrite('./train_noise_5_gradient_result_2_004/gen_intensity_with_noise_%s.bmp'%(sample[idx]), 255*output_intensity[j].eval(feed_dict=feed))
                               cv2.imwrite('./train_noise_5_gradient_result_2_004/gen_phase_with_noise_%s.bmp'%(sample[idx]), 10*output_phase[j].eval(feed_dict=feed))
                               j += 1
                               
                        if learning_rate == 0.0009:
                            for idx in range(len(sample)): 
                               cv2.imwrite('./train_noise_5_gradient_result_2_0009/gen_intensity_with_noise_%s.bmp'%(sample[idx]), 255*output_intensity[j].eval(feed_dict=feed))
                               cv2.imwrite('./train_noise_5_gradient_result_2_0009/gen_phase_with_noise_%s.bmp'%(sample[idx]), 10*output_phase[j].eval(feed_dict=feed))
                               j += 1
                               
                        if learning_rate == 0.0007:
                            for idx in range(len(sample)): 
                               cv2.imwrite('./train_noise_5_gradient_result_2_0007/gen_intensity_with_noise_%s.bmp'%(sample[idx]), 255*output_intensity[j].eval(feed_dict=feed))
                               cv2.imwrite('./train_noise_5_gradient_result_2_0007/gen_phase_with_noise_%s.bmp'%(sample[idx]), 10*output_phase[j].eval(feed_dict=feed))
                               j += 1
                            
                        if flag.coef == 3:
                            for idx in range(len(sample)): 
                               cv2.imwrite('./train_noise_15_gradient_result_3_7epo/gen_intensity_with_noise_%d_%s.bmp'%(noise_level, sample[idx]), 255*output_intensity[j].eval(feed_dict=feed))
                               cv2.imwrite('./train_noise_15_gradient_result_3_7epo/gen_phase_with_noise_%d_%s.bmp'%(noise_level, sample[idx]), 10*output_phase[j].eval(feed_dict=feed))
                               j += 1
                               
                        if train == 0:
                            dir_0 = './train_noise_15_result_add_1_layer'
                            #os.mkdir(dir_0)
                            for idx in range(len(sample)): 
                               cv2.imwrite(dir_0 +'/gen_intensity_with_noise_%d_%s.bmp'%(noise_level, sample[idx]), 255*output_intensity[j].eval(feed_dict=feed))
                               cv2.imwrite(dir_0 +'/gen_phase_with_noise_%d_%s.bmp'%(noise_level, sample[idx]), 10*output_phase[j].eval(feed_dict=feed))
                               j += 1
                        '''
                               
                        if train == 1:
                            dir_1 = './train_noise_0_result_batch1_BNatBegin_newImage'
                            if (tf.gfile.Exists(dir_1)== False):
                                print ('The specified base directory does not exist, now is making directory!')
                                os.mkdir(dir_1)
                                print ('The specified directory is created successfully!')
                            for idx in range(len(sample)): 
                               cv2.imwrite(dir_1+'/coef%d_gen_intensity_with_noise_%d_%s.bmp'%(flag.coef, noise_level, sample[idx]), 255*output_intensity[j].eval(feed_dict=feed))
                               cv2.imwrite(dir_1+'/coef%d_gen_phase_with_noise_%d_%s.bmp'%(flag.coef, noise_level, sample[idx]), 10*output_phase[j].eval(feed_dict=feed))
                               j += 1
                        '''       
                        if train == 2:
                            dir_2 = './train_noise_15_result_del_1_layer_batch1_newImage'
                            for idx in range(len(sample)): 
                               cv2.imwrite(dir_2 +'/gen_intensity_with_noise_%d_%s.bmp'%(noise_level, sample[idx]), 255*output_intensity[j].eval(feed_dict=feed))
                               cv2.imwrite(dir_2 +'/gen_phase_with_noise_%d_%s.bmp'%(noise_level, sample[idx]), 10*output_phase[j].eval(feed_dict=feed))
                               j += 1

                        if flag.coef == 8:
                            for idx in range(len(sample)): 
                               cv2.imwrite('./train_noise_15_gradient_result_8/gen_intensity_with_noise_%d_%s.bmp'%(noise_level, sample[idx]), 255*output_intensity[j].eval(feed_dict=feed))
                               cv2.imwrite('./train_noise_15_gradient_result_8/gen_phase_with_noise_%d_%s.bmp'%(noise_level, sample[idx]), 10*output_phase[j].eval(feed_dict=feed))
                               j += 1
                               
                        if train == 0:
                            for idx in range(len(sample)): 
                               cv2.imwrite('./train_noise_15_gradient_result_0_5/gen_intensity_with_noise_%d_%s.bmp'%(noise_level, sample[idx]), 255*output_intensity[j].eval(feed_dict=feed))
                               cv2.imwrite('./train_noise_15_gradient_result_0_5/gen_phase_with_noise_%d_%s.bmp'%(noise_level, sample[idx]), 10*output_phase[j].eval(feed_dict=feed))
                               j += 1
                               
                        if train == 1:
                            for idx in range(len(sample)): 
                               cv2.imwrite('./train_noise_15_gradient_result_0_5_3epo/gen_intensity_with_noise_%d_%s.bmp'%(noise_level, sample[idx]), 255*output_intensity[j].eval(feed_dict=feed))
                               cv2.imwrite('./train_noise_15_gradient_result_0_5_3epo/gen_phase_with_noise_%d_%s.bmp'%(noise_level, sample[idx]), 10*output_phase[j].eval(feed_dict=feed))
                               j += 1
                         '''
                               
                epoch += 1
            
                
def train_with_gpu(flag):
    pass
