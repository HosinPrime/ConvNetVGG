# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:29:30 2017

@author: xxx
"""

import tensorflow as tf

def conv(layer_name, x, out_channels, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=False):
    print('enter conv')
    
    in_channels = x.get_shape()[-1].value
    
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weights',
                            trainable=is_pretrain,
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer()
                            )
        b = tf.get_variable(name='biases',
                            trainable=is_pretrain,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))
                            
        
        x = tf.nn.conv2d(x, w, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        
        x = tf.nn.relu(x, name='relu')
        
        return x
    

def pool(layer_name, x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True):
    print('enter pool')
    if is_max_pool:
        return tf.nn.max_pool(x, kernel, strides=stride, padding='VALID', name=layer_name)
    else:
        return tf.nn.avg_pool(x, kernel, strides=stride, padding='VALID', name=layer_name)
    

def batch_norm(x):
    print('enter batch_norm')
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x, [0])
    x = tf.nn.batch_normalization(x,
                                  mean=batch_mean,
                                  variance=batch_var,
                                  offset=None,
                                  scale=None,
                                  variance_epsilon=epsilon)
    
    return x


def FC_layer(layer_name, x, out_num):
    print('enter FC_layer')
    shape = x.get_shape()
    
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    
    else:
        size = shape[-1].value
        
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='w', 
                            shape=[size, out_num],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='b',
                            shape=[out_num],
                            initializer=tf.constant_initializer(0.0))
        
        flat_x = tf.reshape(x, [-1, size])
        
        x = tf.nn.bias_add( tf.matmul(flat_x, w), b)
        x = tf.nn.relu(x)
        
        return x
    

def loss(logits, label):
    print('enter loss')
    with tf.name_scope('loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label, name='cross-entropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope+'loss', loss)
        
        return loss
    

def accuracy(logits, labels):
    print('enter accuracy')
    with tf.name_scope('accuracy') as scope:
        corret = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
        corret = tf.cast(corret, tf.float32)
        acc = tf.reduce_mean(corret) * 100
        tf.summary.scalar(scope + 'accuracy', acc)   
        return acc
        
    
def num_correct(logits, labels):
    
    corret = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
    corret = tf.cast(corret, tf.int32)
    
    n = tf.reduce_sum(corret)
    
    return n
    

def optimize(loss, learning_rate, global_step):
    print('enter optimize')
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op
    

def print_all_variables(train_only=True):
    """Print all trainable and non-trainable variables
    without tl.layers.initialize_global_variables(sess)
    Parameters
    ----------
    train_only : boolean
        If True, only print the trainable variables, otherwise, print all variables.
    """
    # tvar = tf.trainable_variables() if train_only else tf.all_variables()
    if train_only:
        t_vars = tf.trainable_variables()
        print("  [*] printing trainable variables")
    else:
        try: # TF1.0
            t_vars = tf.global_variables()
        except: # TF0.12
            t_vars = tf.all_variables()
        print("  [*] printing global variables")
    for idx, v in enumerate(t_vars):
        print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))   
    
    
    
    
    
    
    
    
    
    
    
    