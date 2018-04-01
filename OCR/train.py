import tensorflow as tf
import numpy as np
import os
import sys

import data

#Making Train and Validation Datasets
data_dir = 'Datasets/image_set_train'
data_list = os.listdir(data_dir)
train_data, validation_data, max_len = data.read_data(data_dir, data_list.__len__() - 1, 70, verbose=1)

train_input, train_target, train_lens, num_train = train_data
validation_input, validation_target, validation_lens, num_validation = validation_data

[image_rows, image_cols, image_channels] = train_input[0].shape


#Building Tensorflow Model
tf.reset_default_graph()

inputs = tf.placeholder(tf.float32, [None, image_rows, image_cols, image_channels], name='inputs')
seq_lens = tf.placeholder(tf.int32, [None], name='seq_lens')
targets = tf.sparse_placeholder(tf.int32, name='targets')
maximum_length = tf.constant(max_len, name='max_len')

conv_params = [[64,3,'same',2],[128,3,'same',2],[128,3,'same',2],[max_len,3,'same',2]]
def leaky_relu(x, leakiness=0.0):
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')
with tf.name_scope("convolutions"):
    conv0 = tf.layers.conv2d(inputs,
                             filters = conv_params[0][0],
                             kernel_size = [conv_params[0][1], conv_params[0][1]],
                             padding = conv_params[0][2])
    mean0, variance0 = tf.nn.moments(conv0, [0, 1, 2], keep_dims=False)
    conv0_bn = tf.nn.batch_normalization(conv0, mean0, variance0, offset=0, scale=1, variance_epsilon=0.001)
    conv0_relu = leaky_relu(conv0_bn, 0.01)
    maxpool0 = tf.layers.max_pooling2d(conv0_relu,
                                       pool_size=[conv_params[0][3], conv_params[0][3]],
                                       strides=[conv_params[0][3], conv_params[0][3]])

    conv1 = tf.layers.conv2d(maxpool0,
                             filters = conv_params[1][0],
                             kernel_size = [conv_params[1][1], conv_params[1][1]],
                             padding = conv_params[1][2])
    mean1, variance1 = tf.nn.moments(conv1, [0, 1, 2], keep_dims=False)
    conv1_bn = tf.nn.batch_normalization(conv1, mean1, variance1, offset=0, scale=1, variance_epsilon=0.001)
    conv1_relu = leaky_relu(conv1_bn, 0.01)
    maxpool1 = tf.layers.max_pooling2d(conv1_relu,
                                       pool_size=[conv_params[1][3], conv_params[1][3]],
                                       strides=[conv_params[1][3], conv_params[1][3]])

    conv2 = tf.layers.conv2d(maxpool1,
                             filters = conv_params[2][0],
                             kernel_size = [conv_params[2][1], conv_params[2][1]],
                             padding = conv_params[2][2])
    mean2, variance2 = tf.nn.moments(conv2, [0, 1, 2], keep_dims=False)
    conv2_bn = tf.nn.batch_normalization(conv2, mean2, variance2, offset=0, scale=1, variance_epsilon=0.001)
    conv2_relu = leaky_relu(conv2_bn, 0.01)
    maxpool2 = tf.layers.max_pooling2d(conv2_relu,
                                       pool_size=[conv_params[2][3], conv_params[2][3]],
                                       strides=[conv_params[2][3], conv_params[2][3]])

    conv3 = tf.layers.conv2d(maxpool2,
                             filters = conv_params[3][0],
                             kernel_size = [conv_params[3][1], conv_params[3][1]],
                             padding = conv_params[3][2])
    mean3, variance3 = tf.nn.moments(conv3, [0, 1, 2], keep_dims=False)
    conv3_bn = tf.nn.batch_normalization(conv3, mean3, variance3,  offset=0, scale=1, variance_epsilon=0.001)
    conv3_relu = leaky_relu(conv3_bn, 0.01)
    maxpool3 = tf.layers.max_pooling2d(conv3_relu,
                                       pool_size=[conv_params[3][3], conv_params[3][3]],
                                       strides=[conv_params[3][3], conv_params[3][3]])


    last_shape = maxpool3.get_shape()
    num_last_features = last_shape[1] * last_shape[2]
    seq_features = tf.reshape(maxpool3, [-1,num_last_features,7])

    seq_features = tf.transpose(seq_features, [0, 2, 1])

lstm_units = 64
num_layers = 2
keep_prob = 1
with tf.name_scope("lstm_rnn"):
    def get_a_cell(lstm_size, keep_prob):
        lstm = tf.nn.rnn_cell.LSTMCell(lstm_size)
        drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop

    cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(lstm_units, keep_prob) for _ in range(num_layers)])
    outputs, state = tf.nn.dynamic_rnn(cell=cell, inputs=seq_features, sequence_length=seq_lens, dtype=tf.float32)

charset = '0123456789+-*()'
num_classes = charset.__len__() + 1
W = tf.Variable(tf.random_uniform(shape=[lstm_units, num_classes]))
b = tf.Variable(tf.zeros(shape=[num_classes]))

logits = []
for i in range(max_len):
    logits.append(tf.matmul(tf.reshape(tf.transpose(outputs, [1, 0, 2])[i,:,:], [-1,lstm_units]), W) + b)

logits_tensor = tf.stack(logits)

logits_tensor_batch_major = tf.transpose(logits_tensor, [1, 0, 2])
logits_tensor_unstacked = tf.unstack(logits_tensor_batch_major, axis=1)
output_predicted = [tf.nn.softmax(logits_tensor_unstacked[i]) for i in range(max_len)]

for i in range(max_len):
    tf.add_to_collection("output_colected" + str(i), tf.nn.softmax(logits_tensor_unstacked[i]))

loss = tf.nn.ctc_loss(labels = targets,
                      inputs = logits_tensor,
                      sequence_length=seq_lens)
loss = tf.reduce_mean(loss)
learning_rate = tf.placeholder(tf.float32)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

#Training Procedure
batch_size = 30
num_epochs = 100
lr = 0.01
log_percent = 10
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    train_get_inputs, train_get_labels, train_get_seq_lens = data.get_batch(train_data, 0, np.int(log_percent*num_train/100), max_len)
    train_dict = {inputs: train_get_inputs,
                  targets: train_get_labels,
                  seq_lens: train_get_seq_lens}
    validation_get_inputs, validation_get_labels, validation_get_seq_lens = data.get_batch(validation_data, 0, np.int(log_percent*num_validation/100), max_len)
    validation_dict = {inputs: validation_get_inputs,
                       targets: validation_get_labels,
                       seq_lens: validation_get_seq_lens}

    for epoch in range(num_epochs):
        train_loss = sess.run(loss, feed_dict=train_dict)
        validation_loss = sess.run(loss, feed_dict=validation_dict)

        sys.stdout.write("\r" + "EPOCH: " + str(epoch)
                         + "    Train Loss: " + str(train_loss)
                         + "    Validation Loss: " + str(validation_loss)
                         + "\n")
        sys.stdout.flush()
        for iter, i in enumerate(range(0, num_train - batch_size, batch_size)):
            batch_inputs, batch_labels, batch_lens = data.get_batch(train_data, i, batch_size, max_len)

            batch_dict = {inputs: batch_inputs ,
                          targets: batch_labels ,
                          seq_lens: batch_lens,
                          learning_rate: lr}
            sess.run(optimizer, feed_dict=batch_dict)
            sys.stdout.write("\r" + "processing... " + str(np.floor(i*100/num_train)) + "%")
            sys.stdout.flush()

        saver.save(sess, 'Models/Model')



