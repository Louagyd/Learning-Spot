import tensorflow as tf
import numpy as np
import data
import matplotlib.pyplot as plt
import sys

sess = tf.Session()
saver = tf.train.import_meta_graph('Models/Model.meta')
saver.restore(sess, tf.train.latest_checkpoint('Models'))
graph = tf.get_default_graph()

inputs = graph.get_tensor_by_name('inputs:0')
seq_lens = graph.get_tensor_by_name('seq_lens:0')
max_len = sess.run(graph.get_tensor_by_name('max_len:0'))

output_predicted = [graph.get_collection('output_colected' + str(i)) for i in range(max_len)]
output_predicted = [output_predicted[i][0] for i in range(max_len)]

def predict_images(images, verbose = 0):
    [image_rows, image_cols, image_channels] = images[0].shape
    results = []
    confs = []
    for iter, image_to_predict in enumerate(images):
        if verbose == 1:
            plt.imshow(image_to_predict)
            plt.show()
        image_to_predict = np.reshape(image_to_predict, [1, image_rows, image_cols, image_channels])

        this_output = sess.run(output_predicted, feed_dict={inputs: image_to_predict,
                                                            seq_lens: [7]})
        conf_vec = [max(this_output[i][0]) for i in range(max_len)]
        result = [data.vector_to_char(this_output[i]) for i in range(max_len)]

        if np.sum([int(result[i][0] == '(') for i in range(5)]) == 0:
            result = result[0:5]
            conf_vec = conf_vec[0:5]

        this_conf = np.prod(conf_vec)
        result = ''.join([result[i][0] for i in range(len(result))])

        if verbose == 1:
            print('prediction of image number' + str(iter) + 'is: ' + result + '\n')
            print('confidence vector of predicted characters of result number' + str(iter) + 'is: ' + conf_vec + '\n')
        else:
            sys.stdout.write("\r" + "processing... " + str(np.floor(iter*100/len(images))) + "%")

        confs.append(this_conf)
        results.append(result)
    sys.stdout.write("\r" + "processing... 100%" + "\n")
    return results, confs
