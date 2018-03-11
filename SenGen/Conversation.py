import tensorflow as tf
import gensim
import numpy as np
import random

# loading the generated gensim model
gen_model = gensim.models.Word2Vec.load('GensimModels/gmodel')

# restoring the saved model
sess = tf.Session()
saver = tf.train.import_meta_graph('Models/RNNModel.meta')
saver.restore(sess, tf.train.latest_checkpoint('Models'))
graph = tf.get_default_graph()
input_data = graph.get_tensor_by_name("input_data:0")
max_len = sess.run(graph.get_tensor_by_name("max_len:0"))
input_lens = graph.get_tensor_by_name("input_lens:0")
output = graph.get_collection("output_colected")

edim = gen_model.vector_size

# function for predicting next word. the result is a dict with probability value to be the next word for each word. that is arrenged in descending order.
def next_word(words):
    if (len(words) > max_len):
        words2 = words[-max_len:]
    else:
        words2 = words

    input_vectorized = np.ndarray(dtype=np.float32, shape=(1, max_len, gen_model.vector_size))
    input_vectorized[0, 0:len(words2), :] = gen_model[words2]
    output_predicted = sess.run(tf.reshape(output, [gen_model.vector_size]), feed_dict={input_data: input_vectorized, input_lens: len(words2)})
    output_word = gen_model.similar_by_vector(output_predicted)
    return output_word

# function for generating random with weights. for choosing one word as the predicted word.
def weighted_choice(seq, weights):
    weights = [weights[i]/sum(weights) for i in range(len(seq))]
    assert len(weights) == len(seq)
    assert abs(1. - sum(weights)) < 1e-6

    x = random.random()
    for i, _ in enumerate(seq):
        if x <= weights[i]:
            return seq[i]
        x -= weights[i]

# function for generating sentences started with initial words
def generate_sentence(first_words, len, num_selection):
    all_words = first_words.split()
    for i in range(len):
        values = next_word(all_words)
        possible_words = [values[j][0] for j in range(num_selection)]
        probs = [values[j][1] for j in range(num_selection)]
        the_next_word = weighted_choice(possible_words, probs)
        all_words.append(the_next_word)
    return all_words

# generate some example sentences
sen = generate_sentence("I wish I have a good", 10, 10)
print(sen)
sen = generate_sentence("please do not forget to bring", 10, 10)
print(sen)
sen = generate_sentence("a good business man is always", 10, 10)
print(sen)
sen = generate_sentence("where should I go after", 10, 10)
print(sen)
sen = generate_sentence("there are many requests waiting for", 10, 10)
print(sen)



