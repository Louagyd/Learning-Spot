import os
from collections import Counter
import numpy as np
import random


def read_data(fname, count, word2idx):
    # function for reading data. split whole text into its words and create a dictionary for assigning a number to each word.
    if os.path.isfile(fname):
        with open(fname) as f:
            lines = f.readlines()
    else:
        raise ("[!] Data %s not found" % fname)

    words = []
    for line in lines:
        words.extend(line.split())

    if len(count) == 0:
        count.append(['<eos>', 0])

    count[0][1] += len(lines)
    count.extend(Counter(words).most_common())

    if len(word2idx) == 0:
        word2idx['<eos>'] = 0

    for word, _ in count:
        if word not in word2idx:
            word2idx[word] = len(word2idx)

    data = list()
    for line in lines:
        for word in line.split():
            index = word2idx[word]
            data.append(index)
        data.append(word2idx['<eos>'])

    print("Read %s words from %s" % (len(data), fname))
    return data

def generate_batch(data, index, batch_size, num_skip, skip_window):
    the_index = int(index)
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size,1), dtype=np.int32)
    for i in range(batch_size // num_skip):
        center = data[the_index]
        context_words = [w for w in range(2*skip_window+1) if w != skip_window]
        words_to_use = random.sample(context_words, num_skip)
        for j, word in enumerate(words_to_use):
            batch[i*num_skip + j] = center
            labels[i*num_skip + j] = data[the_index + word - skip_window]
        the_index += 1
    return batch, labels

def read_text(fname, unk = True):

    if unk:
        if os.path.isfile(fname):
            with open(fname) as f:
                lines = f.readlines()
        else:
            raise ("[!] Data %s not found" % fname)
        words = []
        for line in lines:
            words.extend(line.split())

        return words
    else:
        if os.path.isfile(fname):
            with open(fname) as f:
                lines = f.readlines()
        else:
            raise ("[!] Data %s not found" % fname)
        words = []
        for line in lines:
            words.extend(line.replace(" <unk> ", "").replace(" N ", "").replace(" <eof> ", "").replace(" <eoc> ", "").split())

        return words

def text_code(words, word2idx):
    result = []
    for word in words:
        if word in word2idx.keys():
            coded = word2idx[word]
        else:
            coded = word2idx['<unk>']
        result.append(coded)
    return result

def word2vec(words, embedding, word2idx, edim):
    words_coded = text_code(words, word2idx)
    result = np.ndarray(shape=(len(words), edim), dtype=np.float32)
    for i in range(0, len(words_coded)):
        result[i,:] = embedding[words_coded[i]]

    return result
