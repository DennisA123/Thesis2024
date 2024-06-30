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


def entropy(x,y):
    '''
        Calculates the entropy
    '''
    return -torch.sum(x * torch.log(y + 1e-9), dim=1).mean()

class DMDataset(Dataset):
    '''
        Create the dataset for DM training
    '''
    def __init__(self, texts, encodings, labels, vocab, rem):
        self.texts = texts
        self.encodings = encodings
        self.labels = labels
        self.vocab = vocab
        self.rem = rem

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        
        words = self.texts[idx].split()
        words = nltk.word_tokenize(self.texts[idx].lower())
        filtered_words = [word for word in words if word.isalpha() and word not in self.rem]
        
        word_counts = Counter(filtered_words)
        bow_vector = np.zeros(len(self.vocab))
        for word, freq in word_counts.items():
            if word in self.vocab:
                bow_vector[self.vocab.index(word)] = freq / len(filtered_words) if len(filtered_words) > 0 else 0
        item['B'] = bow_vector

        return item

def store_removed_words_list(open_dir, store_dir):
    '''
        Stores gendered and stop words to be removed from the vocab
    '''
    stop_words = stopwords.words('english')
    gender_words = []
    with open(open_dir, 'r') as f:
        word_lines = f.readlines()
    for line in word_lines:
        male_word, female_word = line.strip().split(' - ')
        gender_words.append(male_word)
        gender_words.append(female_word)
    rem = list(set(stop_words + gender_words))
    with open(store_dir, 'w') as json_file:
        json.dump(rem, json_file)

def retrieve_removed_words_list(open_dir):
    '''
        Retrieve words to be removed
    '''
    with open(open_dir, 'r') as json_file:
        retrieved_rem = json.load(json_file)
    return retrieved_rem

def store_vocab(store_dir, texts, retrieved_rem):
    '''
        Store all words from training data - gendered and stop words (=vocab)
    '''
    processed_texts = []
    for text in tqdm(texts):
        words = nltk.word_tokenize(text.lower())
        filtered_words = [word for word in words if word.isalpha() and word not in retrieved_rem]
        processed_texts.extend(filtered_words)
    vocab = list(Counter(processed_texts).keys())
    with open(store_dir, 'w') as json_file:
        json.dump(vocab, json_file)

def retrieve_vocab(open_dir):
    '''
        Retrieve vocab
    '''
    with open(open_dir, 'r') as json_file:
        vocab = json.load(json_file)
    return vocab

def save_plot_all(losses):
    '''
        Plot training losses
    '''
    epochs = range(1, len(losses[0]) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, losses[0], label='D2 correct clf.')
    plt.plot(epochs, losses[1], label='D3 correct clf.')
    plt.plot(epochs, losses[2], label='Reconstruction')
    plt.plot(epochs, losses[3], label='D1 correct clf.')
    plt.plot(epochs, losses[4], label='D2 incorrect clf.')
    plt.plot(epochs, losses[5], label='D3 incorrect clf.')
    plt.plot(epochs, losses[6], label='D4 correct clf.')
    plt.plot(epochs, losses[7], label='Combined')

    plt.title('Training losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./Images/train_loss_curves.png', dpi=300) 
    plt.close()

def prepare_tsne_data(samples, n):
    tsne = TSNE(n_components=2, perplexity=50)
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
    plt.savefig(f'./Images/T_tsne_eval_vis_{name}.png', dpi=300)
    plt.close()