import pickle as pkl
import numpy as np
import tensorflow as tf
import math
from tensorflow.contrib import rnn
import gensim
import json
import jsonlines as js
import numpy as np
import gensimmodel as gm

#Embedding Dimesion
edim = 100

#generate gensim model using the Sentences Dataset
print('generating gensim model...')
pre_sentences_all, pre_sentence_lens_all = gm.generate_model('Dataset/multinli_1.0_train.jsonl', 'gmodel', min_count=0, size=edim, json = True)
# pre_sentences_all, pre_sentence_lens_all = gm.generate_model('Dataset/traintext.txt', 'gmodel', min_count=0, size=edim)

#Load gensim model
gen_model = gensim.models.Word2Vec.load('GensimModels/gmodel')

print('loading and processing data...')
data = js.open('Dataset/multinli_1.0_train.jsonl')
data_train = []
data_valid = []


#filtering the data, only sentences with maximum filter_len words remain
filter_len = 20

data_len = len(pre_sentences_all)

#Biulding train and validation:       -> sentences
#                                     -> sentence_lens
pre_sentences_train = pre_sentences_all[0:np.int(0.7*data_len)]
pre_sentence_lens_train = pre_sentence_lens_all[0:np.int(0.7*data_len)]

pre_sentences_v = pre_sentences_all[np.int(0.7*data_len):data_len]
pre_sentence_lens_v = pre_sentence_lens_all[np.int(0.7*data_len):data_len]

sentences = [pre_sentences_train[i] for i in range(0,len(pre_sentences_train)) if (len(pre_sentences_train[i]) <= filter_len & len(pre_sentences_train[i]) > 1)]
sentence_lens = [pre_sentence_lens_train[i] for i in range(0,len(pre_sentences_train)) if (len(pre_sentences_train[i]) <= filter_len  & len(pre_sentences_train[i]) > 1)]
max_len = np.max(sentence_lens) - 1
print('the train data has ', len(sentences), ' sentences')

sentences_v = [pre_sentences_v[i] for i in range(0,len(pre_sentences_v)) if (len(pre_sentences_v[i]) <= filter_len & len(pre_sentences_v[i]) > 1)]
sentence_lens_v = [pre_sentence_lens_v[i] for i in range(0,len(pre_sentences_v)) if (len(pre_sentences_v[i]) <= filter_len & len(pre_sentences_v[i]) > 1)]
max_len_v = np.max(sentence_lens_v) - 1
print('the validation data has ', len(sentences_v), ' sentences')

print(sentences[0:10])

lstm_units = 128
num_layers = 2
keep_prob = 0.8

#Biulding model with tensorflow
tf.reset_default_graph()
input_data = tf.placeholder(tf.float32, (None, max_len, edim), name="input_data")
input_lens = tf.placeholder(tf.int32, (None), name="input_lens")
tf.add_to_collection("input_colected", input_data)
tf.add_to_collection("input_lens_colected", input_lens)
target_data = tf.placeholder(tf.float32, (None, edim))
learning_rate = tf.placeholder(tf.float32)
maximum_length = tf.constant(max_len, name="max_len")

def get_a_cell(lstm_size, keep_prob):
    lstm = tf.nn.rnn_cell.LSTMCell(lstm_size)
    drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    return drop

cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(lstm_units, keep_prob) for _ in range(num_layers)])
outputs, state = tf.nn.dynamic_rnn(cell=cell, inputs=input_data, sequence_length=input_lens, dtype=tf.float32)

[c, h] = state[-1]
output = tf.layers.dense(h, edim, name="output_data")
tf.add_to_collection("output_colected", output)

loss = tf.reduce_mean(tf.square(output - target_data))
opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

#this function returns a set of datapoints corresponding to every word in sentences
#for example if our sentences are : -> david is going home
#                                   -> I hate mushrooms
#then this function returns: |   input          |    len    |   target     |
#                                david               1            is
#                                david is            2            going
#                                david is going      3            home
#                                I                   1            hate
#                                I hate              2            mushrooms
def get_batch(sentences, seq_lens, index, batch_size, max_len):
    input_seqs = np.zeros(dtype=np.float32, shape=(np.sum(seq_lens[index:(index+batch_size)])-batch_size, max_len, edim))
    input_lens = np.ndarray(dtype=np.int32, shape=(np.sum(seq_lens[index:(index+batch_size)])-batch_size))
    targets = np.ndarray(dtype=np.float32, shape=(np.sum(seq_lens[index:(index+batch_size)])-batch_size, edim))
    ct = 0
    for iter, i in enumerate(range(index, (index+batch_size))):
        this_sentence = sentences[i]
        for j in range(1, seq_lens[i]):
            this_words = this_sentence[0:j]
            this_target = this_sentence[j]
            input_seqs[ct, 0:j, :] = gen_model[this_words]
            input_lens[ct] = j
            targets[ct] = gen_model[this_target]
            ct += 1
    return input_seqs, input_lens, targets


#this function returns a set of datapoints corresponding to every enidng word in sentences
#for example if our sentences are : -> david is going home
#                                   -> I hate mushrooms
#then this function returns: |   input          |    len    |   target     |
#                                david is going      3            home
#                                I hate              2            mushrooms
def get_batch_2(sentences, seq_lens, index, batch_size, max_len):
    input_seqs = np.zeros(dtype=np.float32, shape=(batch_size, max_len, edim))
    input_lens = np.ndarray(dtype=np.int32, shape=(batch_size))
    targets = np.ndarray(dtype=np.float32, shape=(batch_size, edim))
    for iter, i in enumerate(range(index, (index+batch_size))):
        j = seq_lens[i]-1
        this_sentence = sentences[i]
        this_words = this_sentence[0:j]
        this_target = this_sentence[j]
        input_seqs[iter, 0:j, :] = gen_model[this_words]
        input_lens[iter] = j
        targets[iter] = gen_model[this_target]
    return input_seqs, input_lens, targets


#Training Process
sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

train_input_seqs, train_input_lens, train_targets = get_batch(sentences, sentence_lens, 0, len(sentences), max_len)
train_targets_p = [sentences[i][len(sentences[i]) - 1] for i in range(0,len(sentences))]
train_dict = {input_data: train_input_seqs,
              input_lens: train_input_lens,
              target_data: train_targets}

valid_input_seqs, valid_input_lens, valid_targets = get_batch(sentences_v, sentence_lens_v, 0, len(sentences_v), max_len)
valid_targets_p = [sentences_v[i][len(sentences_v[i]) - 1] for i in range(0,len(sentences_v))]
valid_dict = {input_data: valid_input_seqs,
              input_lens: valid_input_lens,
              target_data: valid_targets}

num_epochs = 20
batch_size = 10
lr = 0.001
for epoch in range(0,num_epochs):
    for index in np.arange(0, len(sentences) - batch_size, batch_size):
        batch_input_seqs, batch_input_lens, batch_target = get_batch(sentences, sentence_lens, index, batch_size, max_len)
        # batch_input_seqs, batch_input_lens, batch_target = get_batch_2(sentences, sentence_lens, index, batch_size, max_len)

        batch_dict = {input_data: batch_input_seqs,
                      input_lens: batch_input_lens,
                      target_data: batch_target,
                      learning_rate: lr}
        sess.run(opt, feed_dict=batch_dict)

    print("##########################################EPOCH:", epoch,"#######################################################")
    train_loss = sess.run(loss, feed_dict=train_dict)
    valid_loss = sess.run(loss, feed_dict=valid_dict)
    print('train loss: ', train_loss)
    print('validation loss: ', valid_loss)

# saving the model
saver.save(sess, "Models/RNNModel")
