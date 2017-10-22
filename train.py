# -*- coding: utf-8 -*-

import tensorflow as tf
import Process
import tools
import vgg
import numpy as np
import os

IMG_W = 224
IMG_H = 224
N_CLASSES = 2
BATCH_SIZE = 32
learning_rate = 0.01
MAX_STEP = 15000   # it took me about one hour to complete the training.
IS_PRETRAIN = False
CAPACITY = 1024
train_path = 'F:\\pythonWorkplace\\ConNet\\data\\train'
test_path = 'F:\\pythonWorkplace\\ConNet\\data\\validation'
train_log_dir = 'F:\\pythonWorkplace\\ConNet\\logs\\train'
test_log_dir = 'F:\\pythonWorkplace\\ConNet\\logs\\test'
N_CLASS=2



def train():
    print('loding data............')
    with tf.name_scope('input'):    
        train, train_label, test, test_label = Process.get_data(train_path, test_path)
        train_batch, train_label_batch = Process.get_batch(train, train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
        test_batch, test_label_batch = Process.get_batch(test, test_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
        
    print('loding batch_data complete.......')
    
        
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE, N_CLASS])
    
    
    logits = vgg.VGG16N(x, N_CLASS, IS_PRETRAIN)
    loss = tools.loss(logits, y_)
    accuracy = tools.accuracy(logits, y_)
    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = tools.optimize(loss, learning_rate, my_global_step)
    
    
    saver = tf.train.Saver(tf.global_variables())
    #summary_op = tf.summary.merge_all()
    
    init = tf.global_variables_initializer()
    
    sess = tf.Session()
    sess.run(init)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
  #  tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
  #  val_summary_writer = tf.summary.FileWriter(test_log_dir, sess.graph)
    
    print('all init has been done! start training')
    
    try:
        for step in np.arange(MAX_STEP):
            print('step + ' + str(step) + 'is now')
            if coord.should_stop():
                    break
                
            tra_images,tra_labels = sess.run([train_batch, train_label_batch])
            _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy],
                                            feed_dict={x:tra_images, y_:tra_labels})            
            if step % 10 == 0 or (step + 1) == MAX_STEP:                 
                print ('Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, tra_loss, tra_acc))
               # summary_str = sess.run(summary_op)
               # tra_summary_writer.add_summary(summary_str, step)
                
            if step % 200 == 0 or (step + 1) == MAX_STEP:
                val_images, val_labels = sess.run([test_batch, test_label_batch])
                val_loss, val_acc = sess.run([loss, accuracy],
                                             feed_dict={x:val_images,y_:val_labels})
                print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' %(step, val_loss, val_acc))

              #  summary_str = sess.run(summary_op)
             #   val_summary_writer.add_summary(summary_str, step)
                    
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    