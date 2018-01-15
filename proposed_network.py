from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from ops import layers
from ops import acts
import numpy as np
import re
import random
import os 
from ops import acts
from tensorflow.python.ops import math_ops


class proposed_network(object):
    def __init__(self):
        self.act_fn = acts.pRelu
        self.kernel_num = 16
        self.output_dim = 2
        self.log = 1
        print("FusionNet Loading"),
    
    def Normalize_0_1(self, tensor):
        tensor = tf.div(tf.subtract(tensor, tf.reduce_min(tensor)), tf.subtract(tf.reduce_max(tensor), tf.reduce_min(tensor)))
        #output_255 = tf.scalar_mul(255, output_0_1)
        return tensor
    
        
    def GetIndexList(self, path_in, path_out_intensity, path_out_phase, len_name , batch_size, batch_idx, idx_new):  
        input_ = os.listdir(path_in)
        out_intensity = os.listdir(path_out_intensity)
        out_phase = os.listdir(path_out_phase)
        
        index_in = []
        index_out_intensity = []
        index_out_phase = []
        
        idx = idx_new
        sample = np.random.choice(idx, batch_size, replace=False)
        for i in sample:
            obj_name = input_[i] #2017-07-14_14_07_24_084.bmp
            
            index_in.append(int(i)) 
            index_out_intensity.append(out_intensity.index(obj_name[:-4] + "_bin_gt.bmp"))
            index_out_phase.append(out_phase.index(obj_name[:-4] + "_phase_bin_gt.bmp"))    
        
        sample = list(map(int,sample))

        return index_in, index_out_intensity, index_out_phase, sample
        
    def GetFileList(path):  
        FindPath = path
        FileList = [] 
        FlagStr = 'test' 
        FlagStr = 'jpg'
        FileNames = os.listdir(path)
        if (len(FileNames)>0):  
           for fn in FileNames:  
               if (len(FlagStr)>0):  
                  if (FlagStr in fn):
                      filename= fn 
                      FileList.append(filename)  
                  else:  
               
                      filename= fn 
                      FileList.append(filename)   
        return FileList
        
    def total_variation(self, images, name=None):
         with tf.name_scope('total_variation'):
            ndims = images.get_shape().ndims

            if ndims == 3:
               pixel_dif1 = images[1:, :, :] - images[:-1, :, :]
               pixel_dif2 = images[:, 1:, :] - images[:, :-1, :]

               sum_axis = None
            elif ndims == 4:
               pixel_dif1 = images[:, 1:, :, :] - images[:, :-1, :, :]
               pixel_dif2 = images[:, :, 1:, :] - images[:, :, :-1, :]

               sum_axis = [1, 2, 3]
            else:
               raise ValueError('\'images\' must be either 3 or 4-dimensional.')

            tot_var = (math_ops.reduce_mean(math_ops.abs(pixel_dif1), axis=sum_axis) +
                       math_ops.reduce_mean(math_ops.abs(pixel_dif2), axis=sum_axis))

            return tot_var
                          

    def skip_connection(self, input_, output_):
        return tf.add(input_, output_)

    def res_block_with_n_conv_layers(self, input_, output_dim, num_repeat, name="res_block"):
        output_ = layers.conv2d_same_repeat(input_, output_dim,
                                            num_repeat=num_repeat, activation_fn=self.act_fn, name=name)
        return self.skip_connection(input_, output_)

    def res_block_with_3_conv_layers(self, input_, output_dim, name="res_block"):
        return self.res_block_with_n_conv_layers(input_, output_dim, num_repeat=3, name=name)

    def conv_res_conv_block(self, input_, output_dim, name="conv_res_conv_block"):
        with tf.variable_scope(name):
            conv1 = layers.conv2d_same_act(input_, output_dim, activation_fn=self.act_fn,
                                           with_logit=False, name="conv1")
            res = self.res_block_with_3_conv_layers(conv1, output_dim, name="res_block")
            act = acts.pRelu(res, "act")
            #act = tf.tanh(res)
            conv2 = layers.conv2d_same_act(act, output_dim, activation_fn=self.act_fn,
                                           with_logit=False, name="conv2")
            #return conv2
            return conv2

    def encoder(self, input_):
        self.down1 = self.conv_res_conv_block(input_, self.kernel_num, name="down1")
        pool1 = layers.max_pool(self.down1, name= "pool1")

        self.down2 = self.conv_res_conv_block(pool1, self.kernel_num * 2, name="down2")
        pool2 = layers.max_pool(self.down2, name="pool2")
        
        self.down3 = self.conv_res_conv_block(pool2, self.kernel_num * 4, name="down3")
        pool3 = layers.max_pool(self.down3, name="pool3")

        self.down4 = self.conv_res_conv_block(pool3, self.kernel_num * 8, name="down4")
        pool4 = layers.max_pool(self.down4, name="pool4")

        #self.down5 = self.conv_res_conv_block(pool4, self.kernel_num * 16, name="down5")
        #pool5 = layers.max_pool(self.down5, name="pool5")

            if self.log == 1:
                   print("encoder input : ", input_.get_shape())
                   print("conv1 : ", self.down1.get_shape())
                   print("pool1 : ", pool1.get_shape())
                   print("conv2 : ", self.down2.get_shape())
                   print("pool2 : ", pool2.get_shape())
                   print("conv3 : ", self.down3.get_shape())
                   print("pool3 : ", pool3.get_shape())
                   print("conv4 : ", self.down4.get_shape())
                   print("pool4 : ", pool4.get_shape())
            #print("conv5 : ", self.down5.get_shape())
            #print("pool5 : ", pool5.get_shape())

            return pool4

    def decoder(self, input_):
        #conv_trans5 = layers.conv2dTrans_same_act(input_, self.down5.get_shape(),
                                                # activation_fn=self.act_fn, with_logit=False, name="unpool5")
        #res5 = self.skip_connection(conv_trans5, self.down5)
        #up5 = self.conv_res_conv_block(res5, self.kernel_num * 16, name="up5")
        #with tf.device('/gpu:1'):
        conv_trans4 = layers.conv2dTrans_same_act(input_, self.down4.get_shape(),
                                                  activation_fn=self.act_fn, with_logit=False, name="unpool4")
        res4 = self.skip_connection(conv_trans4, self.down4)
        up4 = self.conv_res_conv_block(res4, self.kernel_num * 8, name="up4")

        conv_trans3 = layers.conv2dTrans_same_act(up4, self.down3.get_shape(),
                                                  activation_fn=self.act_fn, with_logit=False, name="unpool3")
        res3 = self.skip_connection(conv_trans3, self.down3)
        up3 = self.conv_res_conv_block(res3, self.kernel_num * 4, name="up3")

        conv_trans2 = layers.conv2dTrans_same_act(up3, self.down2.get_shape(),
                                                  activation_fn=self.act_fn, with_logit=False, name="unpool2")
        res2 = self.skip_connection(conv_trans2, self.down2)
        up2 = self.conv_res_conv_block(res2, self.kernel_num * 2, name="up2")

        conv_trans1 = layers.conv2dTrans_same_act(up2, self.down1.get_shape(),
                                                  activation_fn=self.act_fn, with_logit=False, name="unpool1")
        res1 = self.skip_connection(conv_trans1, self.down1)
        up1 = self.conv_res_conv_block(res1, self.kernel_num, name="up1")

        if (1 == 1):
            if self.log == 1:
                   print("dncoder input : ", input_.get_shape())
            #print("convT0 : ", conv_trans5.get_shape())
            #print("res0 : ", res5.get_shape())
            #print("up0 : ", up5.get_shape())
                   print("convT1 : ", conv_trans4.get_shape())
                   print("res1 : ", res4.get_shape())
                   print("up1 : ", up4.get_shape())
                   print("convT2 : ", conv_trans3.get_shape())
                   print("res2 : ", res3.get_shape())
                   print("up2 : ", up3.get_shape())
                   print("convT3 : ", conv_trans2.get_shape())
                   print("res3 : ", res2.get_shape())
                   print("up3 : ", up2.get_shape())
                   print("convT4 : ", conv_trans1.get_shape())
                   print("res4 : ", res1.get_shape())
                   print("up4 : ", up1.get_shape())
             return up1

    def inference_encode(self, input_):
        encode_vec = self.encoder(input_)
        
        return encode_vec
        
    def inference_decode(self, encode_vec):
        bridge = self.conv_res_conv_block(encode_vec, self.kernel_num * 16, name="bridge")
        decode_vec = self.decoder(bridge)
        output = layers.bottleneck_layer(decode_vec, self.output_dim, name="output")
        output_0_1 =  self.Normalize_0_1(output)
        output_255 = tf.scalar_mul(255, output_0_1)
        

        if self.log == 1:
            print("output : ", output.get_shape())

        print("Complete!!")

        return  output_255, output_0_1
