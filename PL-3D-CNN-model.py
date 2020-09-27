import numpy as np
import tensorflow as tf


def classifer_3DCNN(feature,training):
        print(feature)
        feature = tf.expand_dims(feature, 4)
        f_num = 16
        with tf.variable_scope('classifer')
            conv0 = tf.layers.conv3d(feature, f_num, (3,3,3), strides=(1,1,1), padding='same')
            conv0 = tf.nn.relu(conv0)
            conv1 = tf.layers.conv3d(conv0, f_num , (3,3,3), strides=(1,1,1), padding='same')
            conv1 = tf.nn.relu(conv1)
            pooling0 = tf.layers.MaxPooling3D((2,2,1), strides=(1,1,1), padding='valid')(conv1)
            pooling0 = tf.nn.relu(pooling0)
            conv2 = tf.layers.conv3d(pooling0, f_num * 2, (3,3,3), strides=(1,1,1), padding='same')
            conv2 = tf.nn.relu(conv2)
            pooling1 = tf.layers.MaxPooling3D((2,2,1), strides=(2,2,1), padding='valid')(conv2)
            pooling1 = tf.nn.relu(pooling1)
            conv3 = tf.layers.conv3d(pooling1, f_num * 2, (3,3,3), strides=(1,1,1), padding='same')
            conv3 = tf.nn.relu(conv3)
            conv4 = tf.layers.conv3d(conv3, f_num * 4, (3,3,3), strides=(1,1,1), padding='valid')
            conv4 = tf.nn.relu(conv4)
            feature = tf.layers.flatten(conv4)
            feature = tf.layers.dropout(feature, 0.5, training=training)
        return feature


def unlabelset_classifier(feature,training,cluster_num):
    with tf.variable_scope('unlabelset_classifier'):
        feature = tf.nn.relu(feature)
        fc = tf.layers.dense(feature,128)
        fc = tf.layers.batch_normalization(fc,training=training)
        fc = tf.nn.relu(fc)
        unlabel_pre_label = tf.layers.dense(fc,cluster_num)
    return fc,unlabel_pre_label

def labelset_classifier(feature,training,class_num):
    with tf.variable_scope('labelset_classifier'):
        feature = tf.nn.relu(feature)
        fc = tf.layers.dense(feature,128)
        fc = tf.layers.batch_normalization(fc,training=training)
        fc = tf.nn.relu(fc)
        labelset_pre_label = tf.layers.dense(fc,class_num)
    return fc,labelset_pre_label

def combine(fc1,fc2,class_num):
    with tf.variable_scope('combine'):
        fusion = tf.concat([fc1,fc2],axis=1)
        logits = tf.layers.dense(fusion,class_num)
    return logits
