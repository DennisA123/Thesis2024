import json
import nltk
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

# def store_removed_words_list(open_dir, store_dir):
#     # gendered and stop words to be removed from the vocab
#     stop_words = stopwords.words('english')
#     gender_words = []
#     # NOTE: this list is very limited --> variations?
#     with open(open_dir, 'r') as f:
#         word_lines = f.readlines()
#     for line in word_lines:
#         male_word, female_word = line.strip().split(' - ')
#         gender_words.append(male_word)
#         gender_words.append(female_word)
#     rem = list(set(stop_words + gender_words))
#     with open(store_dir, 'w') as json_file:
#         json.dump(rem, json_file)

# def retrieve_removed_words_list(open_dir):
#     with open(open_dir, 'r') as json_file:
#         retrieved_rem = json.load(json_file)
#     return retrieved_rem

# def store_vocab(store_dir, texts, retrieved_rem):
#     # get vocab size V for the BoW features
#     processed_texts = []
#     for text in tqdm(texts):
#         words = nltk.word_tokenize(text.lower())
#         # NOTE: keep or remove isalpha?
#         filtered_words = [word for word in words if word.isalpha() and word not in retrieved_rem]
#         processed_texts.extend(filtered_words)
#     vocab = list(Counter(processed_texts).keys())
#     with open(store_dir, 'w') as json_file:
#         json.dump(vocab, json_file)

# def retrieve_vocab(open_dir):
#     with open(open_dir, 'r') as json_file:
#         vocab = json.load(json_file)
#     return vocab

# def pad_x_and_y(token_ids_X, token_ids_Y, mask_X, mask_Y, tokenizer):
#     # assumes pre-padding instead of post-padding!
#     padding_token_id = tokenizer.pad_token_id
#     max_length = max(token_ids_X.size(1), token_ids_Y.size(1))

#     if token_ids_Y.size(1) < max_length:
#         padding_needed = max_length - token_ids_Y.size(1)
#         padding = torch.full((token_ids_Y.size(0), padding_needed), padding_token_id, dtype=token_ids_Y.dtype).to(token_ids_Y.device)
#         token_ids_Y = torch.cat([padding, token_ids_Y], dim=1)
#     elif token_ids_X.size(1) < max_length:
#         padding_needed = max_length - token_ids_X.size(1)
#         padding = torch.full((token_ids_X.size(0), padding_needed), padding_token_id, dtype=token_ids_X.dtype).to(token_ids_X.device)
#         token_ids_X = torch.cat([padding, token_ids_X], dim=1)

#     if mask_Y.size(1) < max_length:
#         padding_needed = max_length - mask_Y.size(1)
#         mask_padding = torch.zeros((mask_Y.size(0), padding_needed), dtype=mask_Y.dtype).to(mask_Y.device)
#         mask_Y = torch.cat([mask_padding, mask_Y], dim=1)
#     elif mask_X.size(1) < max_length:
#         padding_needed = max_length - mask_X.size(1)
#         mask_padding = torch.zeros((mask_X.size(0), padding_needed), dtype=mask_X.dtype).to(mask_X.device)
#         mask_X = torch.cat([mask_padding, mask_X], dim=1)

#     return token_ids_X, token_ids_Y, mask_X, mask_Y

# def get_lengths(batch, eos_id):
#     batch_list = batch.tolist()
#     eos_pos = [[i for i in range(len(ins)) if ins[i] == eos_id] for ins in batch_list]

#     lens = []
#     for ins, eos in zip(batch_list, eos_pos):
#         if len(eos) > 0:
#             lens.append(max(min(eos), 1))
#         else:
#             lens.append(len(ins))

#     return np.asarray(lens)

# def plot_pca(pca_results, labels, title, name=''):
#     plt.figure(figsize=(10, 5))
#     plt.scatter(pca_results[labels == 0, 0], pca_results[labels == 0, 1], color='purple', label='Male')
#     plt.scatter(pca_results[labels == 1, 0], pca_results[labels == 1, 1], color='gold', label='Female')
#     plt.title(title)
#     plt.xlabel('PC1')
#     plt.ylabel('PC2')
#     plt.legend()
#     plt.savefig(f'./imgs/GRU_pca_eval_vis_{name}.png', dpi=300)
#     plt.close()

# def prepare_pca_data(samples, n):
#     pca = PCA(n_components=2)
#     data = torch.cat([samples[0], samples[1]], dim=0)
#     labels = np.array([0]*n + [1]*n)
#     pca_results = pca.fit_transform(data)
#     return pca_results, labels

# ----------------------------------------------------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------------------------------------------------