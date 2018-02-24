import tensorflow as tf
import data
import math
import numpy as np
import pickle

# The number of each word in dictionary. example: {'banana', 1377}
word2idx = {}

# count of each word in text data
count = []

# the training data: Source text split into words and then appear in numbers with dictionary
#example:    Source: David is eating banana
#            trainingdata: [51231, 6182, 78, 1377]
traindata = data.read_data('Dataset/Source.txt', count, word2idx)

vocab_size = len(word2idx)
batch_size = 128

skip_size = 4
skip_window = 2
# for larger skip_size and skip_window, the training process would be much slower

#embedding dimension
edim = 100
num_negative = 64

center = tf.placeholder(tf.int32, [None])
target = tf.placeholder(tf.int32, [None, 1])

embeddings = tf.Variable(tf.random_normal([vocab_size, edim]))
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
embed = tf.nn.embedding_lookup(embeddings, center)

nce_weights = tf.Variable(tf.truncated_normal([vocab_size, edim], stddev=1.0/math.sqrt(edim)))
nce_biases = tf.Variable(tf.zeros(vocab_size))
loss = tf.reduce_mean(
    tf.nn.nce_loss(weights=nce_weights,
                   biases=nce_biases,
                   labels=target,
                   inputs=embed,
                   num_sampled=num_negative,
                   num_classes=vocab_size))
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)


#Training process
train_centers, train_labels = data.generate_batch(traindata, skip_window, batch_size = (len(traindata)- 2*skip_window - 1)*skip_size, num_skip= skip_size, skip_window= skip_window)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_epochs = 50
    for epoch in range(num_epochs):
        for iter, index in enumerate(np.arange(skip_window, len(traindata) - skip_window - batch_size/skip_size, batch_size/skip_size)):
            batches, labels = data.generate_batch(traindata, index, batch_size= batch_size, num_skip= skip_size, skip_window= skip_window)
            batch_dict = {center: batches,
                          target: labels}
            sess.run(optimizer, feed_dict=batch_dict)
        print('EPOCH:   ', epoch+1,'/', num_epochs)
        print(sess.run(loss, feed_dict={center: train_centers,
                                        target: train_labels}))

    final_embeddings_normal = normalized_embeddings.eval()
    final_embeddings = embeddings.eval()
    final_nce_weights = nce_weights.eval()
    final_nce_biases = nce_biases.eval()

#Saving anything
with open('embeddings.pkl', 'wb') as f:
    pickle.dump([final_embeddings_normal, final_embeddings, final_nce_weights, final_nce_biases, word2idx], f)
