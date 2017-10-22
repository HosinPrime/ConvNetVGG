# -*- coding: utf-8 -*-

import tensorflow as tf
import Process
import tools
import vgg
import numpy as np
import os


'''
训练函数
'''


#各种全局所需变量的值设定
#不模型微调整只需要改变这些值就可以了
IMG_W = 224
IMG_H = 224
BATCH_SIZE = 32
learning_rate = 0.01
MAX_STEP = 15000   # 训练的最大步骤数，其数字为大概为total_data/batch_size*epoch左右
IS_PRETRAIN = False
CAPACITY = 1024
train_path = 'D:\\pythonWorkSpace\\ConvNetVGG\\data\\train'
test_path = 'D:\\pythonWorkSpace\\ConvNetVGG\\data\\test'
train_log_dir = 'D:\\pythonWorkSpace\\ConvNetVGG\\logs\\train'
test_log_dir = 'D:\\pythonWorkSpace\\ConvNetVGG\\logs\\test'
N_CLASS=2



def train():
    print('loding data............')

    #导入数据
    with tf.name_scope('input'):    
        train, train_label, test, test_label = Process.get_data(train_path, test_path)
        train_batch, train_label_batch = Process.get_batch(train, train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
        test_batch, test_label_batch = Process.get_batch(test, test_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
        
    print('loding batch_data complete.......')
    
    #创建placeholder作为输入和标签
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE, N_CLASS])
    
    #定义模型
    logits = vgg.VGG16N(x, N_CLASS, IS_PRETRAIN)
    #定义损失
    loss = tools.loss(logits, y_)
    #计算准确率
    accuracy = tools.accuracy(logits, y_)
    #全局步骤
    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    #梯度下降
    train_op = tools.optimize(loss, learning_rate, my_global_step)
    
    #保存训练步骤
    saver = tf.train.Saver(tf.global_variables())
    #summary_op = tf.summary.merge_all()
    #全局变量初始操作
    init = tf.global_variables_initializer()
    #创建sess
    sess = tf.Session()
    #全局变量操作
    sess.run(init)
    #启动coord
    coord = tf.train.Coordinator()
    #启动队列
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  #一些tensorboard的可视化操作，由于会出现问题，我先注释掉了
  #  tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
  #  val_summary_writer = tf.summary.FileWriter(test_log_dir, sess.graph)
    
    print('all init has been done! start training')
    
    try:
        for step in np.arange(MAX_STEP):
            print('step + ' + str(step) + 'is now')
            if coord.should_stop():
                    break
            #从队列中取batch
            tra_images,tra_labels = sess.run([train_batch, train_label_batch])
            #计算损失和准确率
            _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy],
                                            feed_dict={x:tra_images, y_:tra_labels})     

            #如果到达10步的倍数，打印在现在的batch_size上的训练准确率       
            if step % 10 == 0 or (step + 1) == MAX_STEP:                 
                print ('Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, tra_loss, tra_acc))
               # summary_str = sess.run(summary_op)
               # tra_summary_writer.add_summary(summary_str, step)
            
            #如果步骤达到200的倍数，输入一些训练数据查看在训练集上的准确率
            if step % 200 == 0 or (step + 1) == MAX_STEP:
                val_images, val_labels = sess.run([test_batch, test_label_batch])
                val_loss, val_acc = sess.run([loss, accuracy],
                                             feed_dict={x:val_images,y_:val_labels})
                print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' %(step, val_loss, val_acc))

              #  summary_str = sess.run(summary_op)
             #   val_summary_writer.add_summary(summary_str, step)
            
            #如果步骤达到了2000步,保存当前点的数据
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    