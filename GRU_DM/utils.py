import json
import nltk
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

def pad(list, padding=0, min_len=None):
        padded = []
        max_len = max([len(l) for l in list])
        if min_len:
            max_len = max(min_len, max_len)
        for l in list:
            padded.append(l + [padding] * (max_len - len(l)))

        return torch.tensor(padded, dtype=torch.long)

def accuracy(predictions, truths):
    pads = torch.Tensor.double(truths != 0)
    corrects = torch.Tensor.double(predictions == truths)
    valid_corrects = corrects * pads

    return valid_corrects.sum() / pads.sum()

def eval_loss(dists, text, pad_token):
    loss = 0
    num_tokens = 0

    for dist, y in zip(dists, text):
        y_len = sum([1 if y_i != pad_token else 0 for y_i in y])
        for i in range(y_len):
            loss -= torch.log(dist[i][y[i]])
            num_tokens += 1

    return loss, num_tokens

def EntropyLoss(tensor):
    '''
        Calculate the cross entropy loss
    '''
    p = nn.functional.softmax(tensor, dim=1)
    log_p = torch.log(p + 1e-10)
    return torch.mean(torch.sum(p * log_p, dim=1))

def bow_loss(y_bow, bow_labels):
    '''
    :param y_bow: (batch_size x vocab_size)
    :param bow_labels: (batch_size x vocab_size)
    '''
    y_bow = nn.functional.softmax(y_bow, dim=1)
    # loss_for_each_batch: (batch_size)
    loss_for_each_batch = torch.sum(- bow_labels * torch.log(y_bow + 1e-10), dim=1)
    return loss_for_each_batch.mean()

def get_bow(batch, remove_words_id, vocab_size):
    '''
    :param batch: (batch_size x seq_len)
    :return: (batch_size x vocab_size)
    '''
    bows = []
    for text in batch:
        bow = torch.zeros(vocab_size)
        count = {}
        for i in text:
            if i != 0 and i not in remove_words_id:
                if i not in count:
                    count[i] = 1
                else:
                    count[i] += 1
        len = sum([c for w, c in count.items()])
        for w, c in count.items():
            bow[w] = c / len
        bows.append(bow)

    return torch.stack(bows)

def prepare_tsne_data(samples, n):
    tsne = TSNE(n_components=2)
    data = torch.cat([samples[0], samples[1]], dim=0)
    labels = np.array([0]*n + [1]*n)
    tsne_results = tsne.fit_transform(data)
    return tsne_results, labels

def plot_tsne(tsne_results, labels, title, name=''):
    plt.figure(figsize=(10, 5))
    plt.scatter(tsne_results[labels == 0, 0], tsne_results[labels == 0, 1], color='purple', label='Male')
    plt.scatter(tsne_results[labels == 1, 0], tsne_results[labels == 1, 1], color='gold', label='Female')
    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.savefig(f'./Images/GRU_tsne_eval_vis_{name}.png', dpi=300)
    plt.close()