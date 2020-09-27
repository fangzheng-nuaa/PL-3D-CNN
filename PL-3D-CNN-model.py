import numpy as np
import tensorflow as tf


def classifer_3DCNN(feature,training):
        print(feature)
        feature = tf.expand_dims(feature, 4)
        f_num = 16
        with tf.variable_scope('classifer'):
            with tf.variable_scope('conv0'):
                conv0 = tf.layers.conv3d(feature, f_num, (3,3,3), strides=(1,1,1), padding='same')
                conv0 = tf.nn.relu(conv0)
                print(conv0)
            with tf.variable_scope('conv1'):
                conv1 = tf.layers.conv3d(conv0, f_num , (3,3,3), strides=(1,1,1), padding='same')
                conv1 = tf.nn.relu(conv1)
                print(conv1)
            with tf.variable_scope('pooling0'):
                pooling0 = tf.layers.MaxPooling3D((2,2,1), strides=(1,1,1), padding='valid')(conv1)
                pooling0 = tf.nn.relu(pooling0)
                print(pooling0)
            with tf.variable_scope('conv2'):
                conv2 = tf.layers.conv3d(pooling0, f_num * 2, (3,3,3), strides=(1,1,1), padding='same')
                conv2 = tf.nn.relu(conv2)
                print(conv2)
            with tf.variable_scope('pooling1'):
                pooling1 = tf.layers.MaxPooling3D((2,2,1), strides=(2,2,1), padding='valid')(conv2)
                pooling1 = tf.nn.relu(pooling1)
                print(pooling1)
            with tf.variable_scope('conv3'):
                conv3 = tf.layers.conv3d(pooling1, f_num * 2, (3,3,3), strides=(1,1,1), padding='same')
                conv3 = tf.nn.relu(conv3)
                print(conv3)
            with tf.variable_scope('conv4'):
                conv4 = tf.layers.conv3d(conv3, f_num * 4, (3,3,3), strides=(1,1,1), padding='valid')
                conv4 = tf.nn.relu(conv4)
                print(conv4)
            with tf.variable_scope('feature'):
                feature = tf.layers.flatten(conv4)
                feature = tf.layers.dropout(feature, 0.5, training=training)
                print(feature)
        return feature


def unlabelset_classifier(feature,training,cluster_num):
    with tf.variable_scope('unlabelset_classifier'):
        feature = tf.nn.relu(feature)
        with tf.variable_scope('full_connect'):
            fc = tf.layers.dense(feature,128)
            fc = tf.layers.batch_normalization(fc,training=training)
            fc = tf.nn.relu(fc)
        with tf.variable_scope('unlabelset_pre_label'):
            unlabel_pre_label = tf.layers.dense(fc,cluster_num)
    return fc,unlabel_pre_label

def labelset_classifier(feature,training,class_num):
    with tf.variable_scope('labelset_classifier'):
        feature = tf.nn.relu(feature)
        with tf.variable_scope('full_connect'):
            fc = tf.layers.dense(feature,128)
            fc = tf.layers.batch_normalization(fc,training=training)
            fc = tf.nn.relu(fc)
        with tf.variable_scope('labelset_pre_label'):
            labelset_pre_label = tf.layers.dense(fc,class_num)
    return fc,labelset_pre_label

def combine(fc1,fc2,class_num):
    with tf.variable_scope('combine'):
        fusion = tf.concat([fc1,fc2],axis=1)
        with tf.variable_scope('logits'):
            logits = tf.layers.dense(fusion,class_num)
    return logits