#!/Users/majian/anaconda/bin/python
import cv2
import tensorflow as tf
import numpy as np
import os
from resnet import *

if __name__=='__main__':
    #build graph
    name_op = tf.placeholder(dtype = tf.string)
    image_contents = tf.read_file(name_op)
    image = tf.image.decode_jpeg(image_contents, channels = 3)
    image = tf.image.resize_images(image, (FLAGS.img_height, FLAGS.img_width))#need to focus
    image = tf.image.per_image_standardization(image)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image,(-1,FLAGS.img_height, FLAGS.img_width,3))#need to focus
    logits_op = inference(image, reuse=False, is_train=False)
    prob_op = tf.nn.softmax(logits_op)
    #input image
    name = os.path.join('./dataset/model_test/','0a6b4bea357bad4e3943da9780f5856a.jpg')
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('./logs_test_110')
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.Saver()
            saver.restore(sess,ckpt.model_checkpoint_path)
            logits = sess.run(logits_op, feed_dict={name_op:name})
            print('logits:\n', logits)
            prob = sess.run(prob_op ,feed_dict = {name_op:name})[0,:]
            label, prob = np.argmax(prob), np.max(prob)
            print(label,prob)
        else:
            print('no checkpoint found!!!')
