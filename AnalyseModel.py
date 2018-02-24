import pickle as pkl
from  scipy import spatial
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

with open('embeddings.pkl', 'rb') as f:
    final_embeddings_normal, final_embeddings, final_nce_weights, final_nce_biases, word2idx = pkl.load(f)
reversed_dict = dict(zip(word2idx.values(), word2idx.keys()))


def most_similar_to_word(word, embeddings, word2idx):
    reversed_dict = dict(zip(word2idx.values(), word2idx.keys()))
    list = np.arange(0, embeddings.shape[0])
    remlist1 = list[:word2idx[word]]
    remlist2 = list[(word2idx[word]+1):]
    rememb = np.concatenate((embeddings[remlist1], embeddings[remlist2]), axis = 0)
    tree = spatial.KDTree(rememb)
    dist, index = tree.query(embeddings[word2idx[word]])
    if index >= word2idx[word]:
        index += 1
    return dist, reversed_dict[index]

def show_plot(embeddings, word2idx):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    plot_only = 500
    low_dim_embs = tsne.fit_transform(embeddings[:plot_only, :])
    reverse_dictionary = dict(zip(word2idx.values(), word2idx.keys()))
    labels = [reverse_dictionary[i] for i in range(plot_only)]

    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

dist1, word1 = most_similar_to_word('before', final_embeddings, word2idx)
dist2, word2 = most_similar_to_word('prices', final_embeddings, word2idx)
dist3, word3 = most_similar_to_word('adults', final_embeddings, word2idx)
dist4, word4 = most_similar_to_word('month', final_embeddings, word2idx)
dist5, word5 = most_similar_to_word('or', final_embeddings, word2idx)
dist6, word6 = most_similar_to_word('dollar', final_embeddings, word2idx)
dist7, word7 = most_similar_to_word('because', final_embeddings, word2idx)
dist8, word8 = most_similar_to_word('more', final_embeddings, word2idx)
dist9, word9 = most_similar_to_word('below', final_embeddings, word2idx)
dist10, word10 = most_similar_to_word('us', final_embeddings, word2idx)
print("Most similar words:")
print('before -> ', word1)
print('prices -> ', word2)
print('adults -> ', word3)
print('month -> ', word4)
print('or -> ', word5)
print('dollar -> ', word6)
print('because -> ', word7)
print('more -> ', word8)
print('below -> ', word9)
print('us -> ', word10)
input("Press Enter to continue...")

print("Scatter plot of words in a 2-Dimensinal space (using PCA)")
show_plot(final_embeddings, word2idx)
input("Press Enter to continue...")
