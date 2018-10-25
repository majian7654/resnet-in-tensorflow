#!/Users/majian/anaconda/bin/python
import cv2
import tensorflow as tf
import numpy as np
import os
from resnet import *
import json

def multi_test(label_path = './dataset'):
    data_dict = {}
    labels =[]
    label_path = os.path.join(label_path,'ai_challenger_pdr2018_validationset_20180905/AgriculturalDisease_validationset/AgriculturalDisease_validation_annotations.json')
    with open(label_path, 'r') as f:
        label_list = json.load(f)
    for image in label_list:
        data_dict[image['image_id']] = int(image['disease_class'])
  
    total_num = 0
    acc_num = 0
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
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('./logs_test_110')
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.Saver()
            saver.restore(sess,ckpt.model_checkpoint_path)
            for name, label in data_dict.items():
                name = os.path.join('./dataset/ai_challenger_pdr2018_validationset_20180905/AgriculturalDisease_validationset/images', name)
                logits = sess.run(logits_op, feed_dict={name_op:name})
                prob = sess.run(prob_op ,feed_dict = {name_op:name})[0,:]
                pred, prob = np.argmax(prob), np.max(prob)
                total_num += 1
                if pred == label:
                    acc_num +=1
                print(pred) 
            print('acc:',acc_num / total_num)
        else:
            print('no checkpoint found!!!')

if __name__=='__main__':
    multi_test()
