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


# def load_data(file_path):
#     with open(file_path, 'r') as f:
#         data = json.load(f)
#     sentences = [item[0] for item in data]
#     labels = [item[1] for item in data]
#     return sentences, labels

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

# def plot_pca(pca_results, labels, title, name=''):
#     plt.figure(figsize=(10, 5))
#     plt.scatter(pca_results[labels == 0, 0], pca_results[labels == 0, 1], color='blue', label='Male')
#     plt.scatter(pca_results[labels == 1, 0], pca_results[labels == 1, 1], color='red', label='Female')
#     plt.title(title)
#     plt.xlabel('PC1')
#     plt.ylabel('PC2')
#     plt.legend()
#     plt.savefig(f'./imgs/T_pca_eval_vis_{name}.png', dpi=300)
#     plt.close()

# def prepare_pca_data(samples, n):
#     pca = PCA(n_components=2)
#     data = torch.cat([samples[0], samples[1]], dim=0)
#     labels = np.array([0]*n + [1]*n)
#     pca_results = pca.fit_transform(data)
#     return pca_results, labels

def entropy(x,y):
    '''
        Calculates the entropy
    '''
    return -torch.sum(x * torch.log(y + 1e-9), dim=1).mean()

# ----------------------------------------------------------------------------------------------------------------------

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
    plt.savefig(f'./Images/T_tsne_eval_viss_{name}.png', dpi=300)
    plt.close()

# ----------------------------------------------------------------------------------------------------------

# def load_data_gender(data):
#     '''
#         Retrieve texts, responses and genders
#     '''
#     x = [item[0] for item in data]
#     y = [item[1] for item in data]
#     g = [item[2] for item in data]
#     return x, y, g

# def load_data_neutral(data):
#     '''
#         Retrieve texts and responses
#     '''
#     x = [item[0] for item in data]
#     y = [item[1] for item in data]
#     return x, y

# def combine_lists(list1, list2, separator=' '):
#     '''
#         Combine elements of two lists into one, with a separator string in between each pair
#     '''
#     return [f'{a}{separator}{b}' for a, b in zip(list1, list2)]

# class CombinedDataset(Dataset):
#     '''
#         Create dataset of concats of text-response pairs
#     '''
#     def __init__(self, tokenized_texts):
#         self.tokenized_texts = tokenized_texts

#     def __len__(self):
#         return len(self.tokenized_texts['input_ids'])

#     def __getitem__(self, idx):
#         text = {key: val[idx] for key, val in self.tokenized_texts.items()}
#         return text

# class TripletDataset(Dataset):
#     '''
#         Create the gendered dataset for the debiasing loop
#     '''
#     def __init__(self, tokenized_texts, tokenized_responses, genders):
#         self.tokenized_texts = tokenized_texts
#         self.tokenized_responses = tokenized_responses
#         self.genders = genders

#     def __len__(self):
#         return len(self.tokenized_texts['input_ids'])

#     def __getitem__(self, idx):
#         text = {key: val[idx] for key, val in self.tokenized_texts.items()}
#         response = {key: val[idx] for key, val in self.tokenized_responses.items()}
#         gender = self.genders[idx]
#         return text, response, gender
    
# def generate_response(tokenized_prompt, max_generation, attention_mask, model):
#     '''
#         Given a tokenized input prompt, generate the output tokens of the LLM (tokenized)
#     '''
#     output = model.generate(tokenized_prompt, attention_mask=attention_mask, max_new_tokens=max_generation, repetition_penalty=1.2)
#     return output

# def gen_attention_mask(tok_sequence, tokenizer):
#      '''
#         Generate attention mask for a given tokenized sequence
#      '''
#      attention_masks = (tok_sequence != tokenizer.pad_token_id).long()
#      return attention_masks