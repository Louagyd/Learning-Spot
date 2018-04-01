import numpy as np
import matplotlib.image as img
import sys

def read_single_image(dir):
    this_pic = img.imread(dir)
    return this_pic

def read_data(dir, num_data, train_percent, verbose = 0):
    dataset = list()
    for i in range(num_data):
        # 'Datasets/image_set_train/'
        this_pic = img.imread(dir + '/' + str(i) + '.png')
        dataset.append(this_pic)
        if verbose == 1:
            sys.stdout.write('\r' + 'Reading Images... ' + str(np.floor(i*100/num_data)) + '%')
    if verbose == 1:
        sys.stdout.write("\r" + 'Reading Images... ' + '100.0%' + '\n')
    with open(dir + '/labels.txt', 'r') as f:
        lines = f.readlines()
        lines = lines[0:num_data]
        labels = list()
        label_lens = list()
        for i, line in enumerate(lines):
            labels.append(line.split()[0])
            label_lens.append(list(line.split()[0]).__len__())
            if verbose == 1:
                sys.stdout.write("\r" + 'Reading Labels... ' + str(np.floor(i/lines.__len__())) + '%')
    if verbose == 1:
        sys.stdout.write("\r" + 'Reading Labels... ' + '100.0%' + '\n')
    num_train = int(np.floor(train_percent*num_data/100))
    num_validation = num_data - int(np.floor(train_percent*num_data/100))
    train_input = dataset[0:int(np.floor(train_percent*num_data/100))]
    validation_input = dataset[int(np.floor(train_percent*num_data/100)):num_data]
    train_target = labels[0:int(np.floor(train_percent*num_data/100))]
    validation_target = labels[int(np.floor(train_percent*num_data/100)):num_data]
    train_lens = label_lens[0:int(np.floor(train_percent*num_data/100))]
    validation_lens = label_lens[int(np.floor(train_percent*num_data/100)):num_data]
    image_rows = dataset[0].shape[0]
    image_cols = dataset[0].shape[1]
    image_channels = dataset[0].shape[2]
    max_len = max([train_target[i].__len__() for i in range(num_train)])
    train_result = [train_input, train_target, train_lens, num_train]
    validation_result = [validation_input, validation_target, validation_lens, num_validation]
    return train_result, validation_result, max_len

charset = '0123456789+-*()'
encode_maps = {}
decode_maps = {}
for i, char in enumerate(charset):
    encode_maps[char] = i
    decode_maps[i] = char

encode_maps[''] = charset.__len__()
decode_maps[charset.__len__()] = ''

def get_batch(df, index, batch_size, max_len):
    this_inputs, this_labels, this_seq_lens, _ = df
    this_labels = this_labels[index:(index+batch_size)]
    this_inputs = this_inputs[index:(index+batch_size)]
    this_seq_lens = this_seq_lens[index:(index+batch_size)]

    label_indexes = []
    label_values = []

    for i, label in enumerate(this_labels):
        for j, char in enumerate(list(label)):
            label_indexes.append([i, j])
            label_values.append(encode_maps[char])

    label_shape = [this_labels.__len__(), max_len]

    sparse_labels = [label_indexes, label_values, label_shape]
    return np.asarray(this_inputs), sparse_labels, this_seq_lens

def vector_to_char(vectors):
    output_list = [decode_maps[np.argmax(vectors[i,:])] for i in range(vectors.shape[0])]
    return output_list