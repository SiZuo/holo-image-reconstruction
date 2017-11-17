import tensorflow as tf
from FusionNet import FusionNet
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
#from ops import acts
#from ops import layers_bn_after
import os

def test_with_cpu(flag):
    with tf.Graph().as_default():
        #holo = glob('/home/zmxu/dataset/obj/test_holo/*.bmp')
        #intensity = glob('/home/zmxu/dataset/obj/test_intensity/*.bmp')
        #phase = glob('/home/zmxu/dataset/obj/test_phase/*.bmp')
        
        path_holo = '/home/zmxu/dataset/obj/test/test_holo_5'
        path_intensity = '/home/zmxu/dataset/obj/test/test_intensity_5'
        path_phase = '/home/zmxu/dataset/obj/test/test_phase_5'
        
        num_batches_per_epoch = 1 // 1
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0), trainable=False)

        decay_steps = int(num_batches_per_epoch * flag.num_epochs_per_decay)

        lr = tf.train.exponential_decay(flag.initial_learning_rates,
                                        global_step,
                                        decay_steps,
                                        flag.learning_flag.learning_rates_decay_factor,
                                        staircase=True)
                                        

        input_placeholder = tf.placeholder(tf.float32,
                                           [flag.image_height, flag.image_width, flag.channel_dim])
        output_intensity_placeholder = tf.placeholder(tf.float32,
                                           [flag.image_height, flag.image_width, flag.channel_dim])
        output_phase_placeholder = tf.placeholder(tf.float32,
                                           [flag.image_height, flag.image_width, flag.channel_dim])
        
        model = FusionNet()

        input_ = tf.image.resize_images(tf.expand_dims(input_placeholder[:, :, :],0), [flag.image_height//2, flag.image_width//2])
        target_intensity = tf.image.resize_images(tf.expand_dims(output_intensity_placeholder[:, :, :] ,0), [flag.image_height//2, flag.image_width//2])
        target_phase = tf.image.resize_images(tf.expand_dims(output_phase_placeholder[:, :, :],0), [flag.image_height//2, flag.image_width//2])
        
        target_phase_0_1 = model.Normalize_0_1(target_phase)

        with tf.name_scope('summaries'):
             with tf.name_scope('Model'):
                with tf.device('/gpu:0'):
                     encode = model.inference_encode(input_)
                with tf.device('/gpu:1'):
                     output_0_255, output_0_1 = model.inference_decode(encode)
                
                output_intensity, output_phase = tf.split(output_0_1, [1,1], 3) 
                target_0_225 = tf.concat([target_intensity, 255*target_phase_0_1], 3)
                target_0_1 = model.Normalize_0_1(target_0_225)
              
         
             optimizer = tf.train.AdamOptimizer(lr)
             
             #sharpen_image = tf.image.adjust_contrast(output_0_1, 1.5)
             #loss = tf.losses.mean_squared_error(target_0_1, sharpen_image)
             loss = tf.losses.mean_squared_error(target_0_1, output_0_1) 
        
        var_list = tf.trainable_variables()

        
        grads_and_vars = optimizer.compute_gradients(loss, var_list=var_list)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)

        init = tf.initialize_all_variables()

        config = tf.ConfigProto()
        config.allow_soft_placement=True
        config.gpu_options.allow_growth = True
        #config.log_device_placement = True
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
       

        with tf.Session(config=config) as sess:
            
            sess.run(init)
            print("Learning start!!")
            start = time.time()

           
            #add noise
            noise = np.zeros((flag.image_height, flag.image_width, 1), np.int8)
            #for rates in flag.initial_learning_rates:
            #for coef in flag.coef_total:
            
            holo_name_list = os.listdir(path_holo)
            for coef in flag.coef:
              if utils.load_ckpt(flag.ckpt_dir, sess, flag.ckpt_name):
                 print(" [*] Load SUCCESS")
              else:
                 print(" [!] Load failed...")
              print ("------------------------------------")
              
              holo_name_list_10 = holo_name_list[:5]
              for name in holo_name_list_10:
               #  in_ind.append(data_test_namelist.index("test_holo" + str(num) + ".bmp")) 
                # out_intensity_ind.append(data_test_namelist.index("test_ori" + str(num) + ".bmp"))
                 #out_phase_ind.append(data_test_namelist.index("test_ori" + str(num) + "_phase.bmp"))
              
                 print ("Image name is: %s"%name)
                
                 for noise_level in flag.noise_total:
                    noises = cv2.randn(noise, np.zeros(1), np.ones(1)*noise_level)
                    batch_in_image = (cv2.imread(path_holo + '/' + name )+ noises)
                    batch_in_images = np.array(batch_in_image).astype(np.float32)[ :, :, :1]
                    
                    #batch_out_intensity_files = [data_test[idx] for idx in out_intensity_ind]
                    #batch_out_intensity_image = [(cv2.imread(batch_out_intensity_file)+ noises) for batch_out_intensity_file in batch_out_intensity_files]
                    
                    intensity_name =  name[:-4] + '_bin_gt.bmp'
                    batch_out_intensity_image = cv2.imread(path_intensity + '/' + intensity_name)
                    batch_out_intensity_images = np.array(batch_out_intensity_image).astype(np.float32)[ :, :, :1]
                    #batch_out_intensity_images = np.array(batch_out_intensity_image).astype(np.float32)[ :, :, :, :1]
                    
                    
                    #batch_out_phase_files = [data_test[idx] for idx in out_phase_ind]
                    #batch_out_phase_image = [(cv2.imread(batch_out_phase_file)+ noises) for batch_out_phase_file in batch_out_phase_files]
                    
                    phase_name = name[:-4] + '_phase_bin_gt.bmp'
                    batch_out_phase_image = cv2.imread(path_phase + '/' + phase_name)
                    batch_out_phase_images = np.array(batch_out_phase_image).astype(np.float32)[ :, :, :1]
                    #batch_out_phase_images = np.array(batch_out_phase_image).astype(np.float32)[ :, :, :, :1]
                    
           
                    feed = {input_placeholder: batch_in_images, output_intensity_placeholder: batch_out_intensity_images, output_phase_placeholder: batch_out_phase_images}
            
                    print("(%ds) noise_level: %d, loss: %f" % (time.time() - start, noise_level, sess.run(loss, feed)))
                    #print("(%ds) Image number: %d, noise_level: %d, loss: %f, regu_loss: %f" % (time.time() - start, num, noise_level, sess.run(loss, feed), sess.run(regu_loss, feed)))
                        
                    epo = 62002
                    
                    #if (coef == 0):
                    #####different epoch
                      # j = 0
                       #for idx in in_ind:
                        #    image_name=data_test_namelist[idx]
                         #   image_name = int(image_name[9:-4])
                          #  cv2.imwrite('./predict/intensity_phase_with_noise_15_del_1/batch10_%d_input_%d_with_noise_%d.bmp'%(epo, image_name, noise_level) ,input_[j].eval(feed_dict=feed))
                           # cv2.imwrite('./predict/intensity_phase_with_noise_15_del_1/batch10_%d_gen_intensity_%d_with_noise_%d.bmp'%( epo, image_name, noise_level),255*output_intensity[j].eval(feed_dict=feed))
                            #cv2.imwrite('./predict/intensity_phase_with_noise_15_del_1/batch10_%d_gen_phase_%d_with_noise_%d.bmp'%( epo, image_name, noise_level),10*output_phase[j].eval(feed_dict=feed))
                            #cv2.imwrite('./predict/intensity_phase_with_noise_15_del_1/batch10_%d_gen_phase_%d_100_with_noise_%d.bmp'%( epo, image_name, noise_level), 100*output_phase[j].eval(feed_dict=feed))
                            #j +=1
                         
                    if (coef == 1):
                    #####different epoch
                       
                       cv2.imwrite('./predict/intensity_phase_with_noise_15_newImage/coef0_%d_input_%s_noise_%d.bmp'%(epo, name[:-4], noise_level) ,input_[0].eval(feed_dict=feed))
                       cv2.imwrite('./predict/intensity_phase_with_noise_15_newImage/coef0_%d_target_%s_noise_%d.bmp'%(epo, name[:-4], noise_level) ,target_intensity[0].eval(feed_dict=feed))
                       cv2.imwrite('./predict/intensity_phase_with_noise_15_newImage/coef0_%d_intensity_%s_noise_%d.bmp'%(epo, intensity_name[:-11], noise_level),255*output_intensity[0].eval(feed_dict=feed))
                      # cv2.imwrite('./predict/intensity_phase_with_noise_15_newImage_BnAfterAct/coef1_%d_phase_%s_noise_%d.bmp'%( epo, phase_name[:-4], noise_level),10*output_phase[0].eval(feed_dict=feed))
                       cv2.imwrite('./predict/intensity_phase_with_noise_15_newImage/coef0_%d_100phase_%s_noise_%d.bmp'%( epo, phase_name[:-17], noise_level), 100*output_phase[0].eval(feed_dict=feed))
                         
                    
                    ######different checkpoint
                    #if ((coef == 3 ) or (coef ==0)):
                      # if (coef == 3):
                    #if (coef == 3):
                       #epo = 8802
                       #else:
                        # epo = 9002
                     #  cv2.imwrite('./predict_with_gradient/regularization_%d_new/intensity_phase_with_noise_15/%d_input_%d_with_noise_%d.bmp'%( coef, epo, num, noise_level) ,input_[0].eval(feed_dict=feed))
                      # cv2.imwrite('./predict_with_gradient/regularization_%d_new/intensity_phase_with_noise_15/%d_gen_intensity_%d_with_noise_%d.bmp'%( coef, epo, num, noise_level),255*output_intensity[0].eval(feed_dict=feed))
                     #  cv2.imwrite('./predict_with_gradient/regularization_%d_new/intensity_phase_with_noise_15/%d_gen_phase_%d_with_noise_%d.bmp'%( coef, epo, num, noise_level),10*output_phase[0].eval(feed_dict=feed))
                     #  cv2.imwrite('./predict_with_gradient/regularization_%d_new/intensity_phase_with_noise_15/%d_gen_phase_%d_100_with_noise_%d.bmp'%( coef, epo, num, noise_level), 100*output_phase[0].eval(feed_dict=feed))
                   
                    
                    
#change checkpoint
              #if coef == 35:
               #     flag.ckpt_dir = './checkpoint_with_noise_15_gradient_3_2epo'     
              #if coef == 32:
               #     flag.ckpt_dir = './checkpoint_with_noise_15'
              #if rates == 0.003:
               #     flag.ckpt_dir = './checkpoint_with_noise_5_gradient_2_004'
              #if rates == 0.004:
               #     flag.ckpt_dir = './checkpoint_with_noise_5_gradient_2_005'
                    
              #if rates == 0.002:
               #     flag.ckpt_dir = './checkpoint_with_noise_5_gradient_2_0009'    
                   
              #if rate == 0.0007:
               #     flag.ckpt_dir = './checkpoint_with_noise_5_gradient_2_0005'            
              
          
                           
                    
      