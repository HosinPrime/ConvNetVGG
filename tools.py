# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:29:30 2017

@author: liuhaoxin
"""
'''
此文件为工具类，包括：
                 1.卷积操作
                 2.池化操作
                 3.归一化
                 4.全连接层
                 5.定义损失函数
                 6.定义优化函数
                 7.测试准确率函数
                 8.测试准确数操作

'''

import tensorflow as tf

def conv(layer_name, x, out_channels, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=False):
    '''
        卷积操作

        输入参数为:
                layter_name:卷积层名字
                x:输入的图片数据
                out_channels:输出的层数
                kernel_size:卷积核的大小
                stride:步伐大小
                is_pretrain:是否事先导入已经训练好的参数
        输出参数为:
                x:输出卷积后的结果

    '''



    #得到输入的通道数
    in_channels = x.get_shape()[-1].value
    
    with tf.variable_scope(layer_name):
        #定义weight参数
        w = tf.get_variable(name='weights',
                            trainable=is_pretrain,
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer()
                            )

        #定义bias参数
        b = tf.get_variable(name='biases',
                            trainable=is_pretrain,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))
                            
        #进行卷积操作并加加上bias
        x = tf.nn.conv2d(x, w, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        
        #relu操作
        x = tf.nn.relu(x, name='relu')
        
        return x
    
#池化操作
def pool(layer_name, x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True):
    '''
    输入参数：
            layter_name:池化层的名字
            x:输入的数据
            kernel:核大小
            stride:步伐
            is_max_pool:是否是最大池化

    输出参数:
            池化后的参数
    '''
    if is_max_pool:
        return tf.nn.max_pool(x, kernel, strides=stride, padding='VALID', name=layer_name)
    else:
        return tf.nn.avg_pool(x, kernel, strides=stride, padding='VALID', name=layer_name)
    
#此函数在vgg16模型中没有用到
def batch_norm(x):
    '''
    输入参数：
            x:输入数据
    输出参数:
            归一化后的数据
    '''

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
    '''
    输入参数:
            layer_name:全连接层的名字
            x:输入数据
    输出参数:
            x:全连接后的得到的输出
    '''
    shape = x.get_shape()
    

    #如果图片还没有展开为vector的形式，得到它展开后的维度
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    
    #否则世界得到大小
    else:
        size = shape[-1].value
        
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='w', 
                            shape=[size, out_num],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='b',
                            shape=[out_num],
                            initializer=tf.constant_initializer(0.0))
        
        #展开为向量模式，-1代表此维度大小不变
        flat_x = tf.reshape(x, [-1, size])
        

        #进行矩阵w*x+b运算并relu
        x = tf.nn.bias_add( tf.matmul(flat_x, w), b)
        x = tf.nn.relu(x)
        
        return x
    

def loss(logits, label):
    '''
    损失函数
    输入参数:
            logits:预测的值
            label:真实的值
    输出参数:
            互信息熵损失
    '''
    with tf.name_scope('loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label, name='cross-entropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        
        return loss
    

def accuracy(logits, labels):
    '''
    测试准确率
    输入参数:
            logits:预测的值
            label:真实的值
    输出参数:
            准确率

    '''
    with tf.name_scope('accuracy') as scope:
        corret = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
        corret = tf.cast(corret, tf.float32)
        acc = tf.reduce_mean(corret) * 100
       # tf.summary.scalar(scope + 'accuracy', acc)   
        return acc
        
    
def num_correct(logits, labels):
    '''
    测试正确预测的数量
    输入参数:
            logits:预测的值
            label:真实的值
    输出参数:
            预测正确的数量
    '''
    corret = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
    corret = tf.cast(corret, tf.int32)
    
    n = tf.reduce_sum(corret)
    
    return n
    

def optimize(loss, learning_rate, global_step):
    '''
    优化函数
    输入参数:
            loss:损失函数
            learning_rate: 学习速率
            global_step:全局步骤
    输出参数:
            train_op:训练操作

    '''

    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op
    
    
    
    
    
    
    
    
    
    
    
    
    