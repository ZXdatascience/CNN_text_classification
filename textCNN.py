import tensorflow as tf
import numpy as np

class TextCNN(object):
    
    def __init__(
        self, sequence_length, num_classes, vacab_size, embedding_size, \
        filter_sizes, num_filters):
        
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name= "input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name= "input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name= "dropout_keep_prob")
    
        with tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name= "W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

#################################    
##construct the convolutional and maxpool layer
#################################
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev= 0.1), name= "W")
                b = tf.Variable(tf.constant(0.1, shape= [num_filters]), name= "b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides= [1,1,1,1],
                    padding= "VALID",
                    name = "conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name= "relu")
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size +1, 1, 1],
                    strides= [1,1,1,1],
                pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
#################################    
##construct the dropout layer
#################################
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
#################################    
##construct the output layer
#################################
        with tf.name_scope("output"):
        W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
        self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
        self.predictions = tf.argmax(self.scores, 1, name="predictions")
#################################    
##calculate loss and accuracy
#################################
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses)
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name= "accuracy")

