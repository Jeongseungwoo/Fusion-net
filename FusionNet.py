from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.layers import Conv2D, Conv2DTranspose
from tensorflow.estimator import *
import numpy as np

import os

tf.logging.set_verbosity(tf.logging.INFO)

def Network(features, labels, mode) :
    
    # input shape : 640 x 640 x 1 
    input_layer = tf.reshape(features["x"],[-1,640,640,1])
    
    # encoder
    down1 = MaxPooling2D([2,2],2)(conv_res_conv(input_layer,64))
    down2 = MaxPooling2D([2,2],2)(conv_res_conv(down1,128))
    down3 = MaxPooling2D([2,2],2)(conv_res_conv(down2,256))
    down4 = MaxPooling2D([2,2],2)(conv_res_conv(down3,512))
    
    # bridge 
    bridge = conv_res_conv(down4,1024)
    
    # decoder
    deconv4 = deconv2d_with_bn_and_act(bridge,down4.get_shape())
    merge4 = skip_connection(deconv4,down4)
    upscale4 = conv_res_conv(merge4,512)
    
    deconv3 = deconv2d_with_bn_and_act(upscale4,down3.get_shape())
    merge3 = skip_connection(deconv3,down3)
    upscale3 = conv_res_conv(merge3,256)
    
    deconv2 = deconv2d_with_bn_and_act(upscale3,down2.get_shape())
    merge2 = skip_connection(deconv2,down2)
    upscale2 = conv_res_conv(merge2,128)
    
    deconv1 = deconv2d_with_bn_and_act(upscale2,down1.get_shape())
    merge1 = skip_connection(deconv1,down1)
    upscale1 = conv_res_conv(merge1,64)
    
    output = Conv2D(1,kernel_size=[1,1],padding="same")(upscale1)
    
    predictions = { "outputs" : output }
    
    if mode == ModeKeys.PREDICT : 
        return EstimatorSpec(mode=mode,predictions=predictions)
    
    loss = tf.losses.mean_pairwise_squared_error(labels,predictions)
    
    if mode == ModeKeys.TRAIN : 
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize (
            loss = loss, 
            global_step = tf.train.get_global_step())
        return EstimatorSpec(mode=mode,loss=loss,train_op=train_op)
    
    eval_metric_ops = {
        "acc" : tf.metrics.accuracy(labels=labels,predictions=predictions["outputs"])
    }
    
    return EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_metric_ops)


##################################################
#                     Layers                     #
##################################################

def skip_connection(input_1, input_2) :
    return tf.add(input_1,input_2)

def conv2d_with_bn_and_act(input_,filters,k_size=[3,3],strides=[1,1],padding="same") :
    return BatchNormalization()(Conv2D(filters,kernel_size=k_size,strides=strides,activation=tf.nn.relu)(input_))

def conv2d_with_3_layers(input_,filters,kernel_size=[3,3],stride=[1,1], repeat = 3) :
    for i in range(repeat) :
        conv = conv2d_with_bn_and_act(input_=input_,filters=filters)
        input_ = conv
    return conv

def res_block(input_,filters) :
    conv = conv2d_with_3_layers(input_,filters)
    return skip_connection(input_,conv)
        
def conv_res_conv(input_,filters) : 
    conv1 = conv2d_with_bn_and_act(input_,filters)
    res = res_block(conv1,filters)
    conv2 = conv2d_with_bn_and_act(res, filters)
    return conv2

def deconv2d_with_bn_and_act(input_,filters,k_size=[3,3],strides=[1,1],padding="same") :
    return BatchNormalization()(Conv2DTranspose(filters,kernel_size=k_size,strides=strides,activation=tf.nn.relu)(input_))

##################################################

# config
batch_size = 100, model_path = "./model", steps = 10000

def main(usused_argv) : 
    # Load your dataset 
    train_data = None 
    train_labels = None 
    eval_data = None 
    eval_labels = None
    
    model_path = model_path
    if not os.path.exists(model_path) :
        os.mkdir(model_path)
    # create the estimator 
    fusionnet = Estimator(model_fn = Network,model_dir=model_path)
    
    train_input = inputs.numpy_input_fn(
        x = {"x" : train_data},
        y = train_labels,
        batch_size = batch_size,
        num_epochs = None,
        shuffle = True)
    
    fusionnet.train(
        input_fn=train_input,
        steps = steps)
    
    eval_input = inputs.numpy_input_fn(
         x = {"x" : eval_data},
         y = eval_labels,
         num_epochs = 1,
         shuffle = False)
    eval_results = fusionnet.evaluate(input_fn = eval_input)
    print(eval_results)

    
if __name__ = "__main__" : 
    tf.app.run()


