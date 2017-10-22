# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import os
import numpy as np
import tensorflow as tf

'''
train_path = 'D:\\pythonWorkSpace\\ConvNetVGG\\data\\train'
test_path = 'D:\\pythonWorkSpace\\ConvNetVGG\\data\\test'


IMAGE_W = 224
IMAGE_H = 224
BATCH_SIZE = 16
CAPACITY = 1024
'''




def get_data(train_filePath, test_filePath):
    '''
    输入参数：
            train_filePath: 训练数据的存储路径,比如F:\\pythonWorkplace\\ConNet\\data\\train
            test_filePath: 测试数据的存储路径 
    返回参数：
            train_data: 训练图像的存储路径,比如[F:\\pythonWorkplace\\ConNet\\data\\train\\1.jpg, F:\\pythonWorkplace\\ConNet\\data\\train\\2.jpg]
            train_label: 训练图像的类别标签,比如[0,1]
            test_data:测试数据的存储路径
            test_label:测试数据的列别标签
            
            上面四个返回参数都是np.array()格式
    '''
    
    
    #每种数据各自的路径和类别标签
    train_data = []                     
    train_label = []
    test_data = []
    test_label = []
    for file in os.listdir(train_filePath):
        image_path = train_filePath + '\\' + file
        for image in os.listdir(image_path):
            im = image_path + '\\' + image
            train_data.append(im)
            if file == 'cats':
                train_label.append(0)
            elif file == 'dogs':
                train_label.append(1)
                
    for file in os.listdir(test_filePath):
        image_path = test_filePath + '\\' + file
        for image in os.listdir(image_path):
            im = image_path + '\\' + image
            test_data.append(im)
            if file == 'cats':
                test_label.append(0)
            elif file == 'dogs':
                test_label.append(1)
                
    '''
    这部分将训练数据和标签堆叠，并随机打乱
    '''
    train = np.array([train_data, train_label])
    train = train.T                  #train的shape是(样本数，2),第一列为样本，第二列为标签
    np.random.shuffle(train)         #shuffle是在不同行之间shuffle所以上面有一个取转置的操作
    train_data = train[:, 0]
    train_label = train[:, 1]
    
    #同理
    test = np.array([test_data, test_label])
    test = test.T
    np.random.shuffle(test)
    test_data = test[:, 0]
    test_label = test[:, 1]
    
    #将标签的类型确定为int
    train_label = train_label.astype(np.int32)
    test_label = test_label.astype(np.int32)
    
    return train_data, train_label, test_data, test_label



def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    输入参数：
            image:上面get_data()得到的image数据路劲列表，可输入训练集或者测试集
            label:上面得到的label
            image_W:希望输入训练的图片宽,该函数会进行裁剪
            image_H:希望输入训练的图片高
            batch_size:每次训练batch_size
            capacity:输入队列的容量
            
    返回参数：
            image_batch:图片batch,大小为（batch_size,image_W,image_H,3）,应该大部分是这样
            label_batch:对应的标签的one-hot编码格式
            
    '''
    
    #将路径列表和标签转换为tensor形式
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
    
    #创建一个输入队列，将数据周期性的输入到内存中
    input_queue = tf.train.slice_input_producer([image, label])
    
    label = input_queue[1]

    #读取文件
    image_content = tf.read_file(input_queue[0])
    #将jpg图片decode为RGB三位形式
    image = tf.image.decode_jpeg(image_content, channels=3)
    
    #将图片resize到用户输入的大小
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    
    #将图片进行标准化操作
    image = tf.image.per_image_standardization(image)

    
    #得到输入的batchsize大小
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=2,
                                              capacity=capacity)
    
    #将图片数据转为float32
    image_batch = tf.cast(image_batch, tf.float32)
    

    #进行one-hot编码，现在要分两类所以类别是2，以后要多分类需要自己修改
    n_classes = 2
    #进行onehot并该形式为int32,onehot后形式为[0,1] [1,0]等相似形式
    label_batch = tf.one_hot(label_batch, depth= n_classes)
    label_batch = tf.cast(label_batch, dtype=tf.int32)
    label_batch = tf.reshape(label_batch, [batch_size, n_classes])
    
    
    return image_batch, label_batch




'''
下面为测试图片导入是否成功的代码，需要使用时直接将注释去掉(并注释掉上面的图片标准化,和转为float的操作)，运行这个文件就可以
'''


'''
import matplotlib.pyplot as plt

BATCH_SIZE = 2
CAPACITY = 256
IMG_W = 224
IMG_H = 224

train_dir = 'D:\\pythonWorkSpace\\ConvNetVGG\\data\\train'
test_dir = 'D:\\pythonWorkSpace\\ConvNetVGG\\data\\test'

image_list, label_list, test_list, test_label = get_data(train_dir,test_dir)
image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

with tf.Session() as sess:
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    try:
        while not coord.should_stop() and i<1:
            
            img, label = sess.run([image_batch, label_batch])
            
            #只测试一个batch
            for j in np.arange(BATCH_SIZE):
                print(label[j])
                plt.imshow(img[j,:,:,:])
                plt.show()
            i+=1
            
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)
'''
