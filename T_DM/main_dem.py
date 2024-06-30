from utils import *
from dem import *
from train_dem import *
from eval_dem import eval_dem_trained
import random

from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BloomTokenizerFast
# nltk.download('punkt')

import torch
import matplotlib.pyplot as plt
import sys
torch.cuda.empty_cache()
import argparse
from torch.utils.data import DataLoader

# e.g:
# python ./T_DM/main_dem.py --save_dem --verbose --what train
# python ./T_DM/main_dem.py --what eval
parser = argparse.ArgumentParser()
parser.add_argument('--store_data', action='store_true', help='Create the necessary files if they do not already exist')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizers')
parser.add_argument('--save_dem', action='store_true', help='Save the disentanglement model after training')
parser.add_argument('--verbose', action='store_true', help='Print losses')
parser.add_argument('--k_values', nargs=5, type=int, metavar=('k0', 'k1', 'k2', 'k3', 'k4'), help='Hyperparams for loss')
parser.add_argument('--u', type=int, default=500, help='unbiased gender vector dimension')
parser.add_argument('--s', type=int, default=500, help='semantic vector dimension')
parser.add_argument('--what', choices=['train', 'eval'], help='Whether to train or evaluate the DM')
parser.add_argument('--dem_name', type=str, default='./Models/dem_bloom_0.001.pth')
parser.add_argument('--D1_name', type=str, default='./Models/D1_bloom_0.001.pth')
parser.add_argument('--D2_name', type=str, default='./Models/D2_bloom_0.001.pth')
parser.add_argument('--n_visualize', type=int, default=250, help='Number of datapoints per gender to visualize with t-SNE')

args = parser.parse_args()

if __name__ == "__main__":

    SEQLEN = 19
    random.seed(8)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # BLOOM TOKENIZER: padding: 3, attention mask on padding: 0, attention mask on rest: 1, vocab size: 250 680
    tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")

    with open(f'./T_DM/Data/unbiased_corpus.json', 'r') as f:
        data_list = json.load(f)
        label_0 = [datapoint for datapoint in data_list if datapoint[1] == 0]
        label_1 = [datapoint for datapoint in data_list if datapoint[1] == 1]
        # shuffle such that random data is selected when only taking first min_count
        random.shuffle(label_0)
        random.shuffle(label_1)
        min_count = min(len(label_0), len(label_1))
        balanced_label_0 = label_0[:min_count]
        balanced_label_1 = label_1[:min_count]
        balanced_data_list = balanced_label_0 + balanced_label_1
        train_data_list, both = train_test_split(balanced_data_list, test_size=0.2, random_state=8)
        valid_data_list, test_data_list = train_test_split(both, test_size=0.5, random_state=8)

    train_texts = [a[0] for a in train_data_list]
    train_labels = [a[1] for a in train_data_list]

    train_texts_19 = []
    train_labels_19 = []
    # only consider data of length 19
    for text, label in zip(train_texts, train_labels):
        tokens = tokenizer(text)
        if len(tokens['input_ids']) == SEQLEN:
            train_texts_19.append(text)
            train_labels_19.append(label)

    test_texts = [a[0] for a in test_data_list]
    test_labels = [a[1] for a in test_data_list]

    test_texts_19 = []
    test_labels_19 = []
    # only consider data of length 19
    for text, label in zip(test_texts, test_labels):
        tokens = tokenizer(text)
        if len(tokens['input_ids']) == SEQLEN:
            test_texts_19.append(text)
            test_labels_19.append(label)

    # words to be removed from the vocab (gendered words + stop words)
    if args.store_data:
        store_removed_words_list(open_dir='./T_DM/Data/gender_words.txt', store_dir='./T_DM/Data/rem_list.json')
    retrieved_rem = retrieve_removed_words_list(open_dir='./T_DM/Data/rem_list.json')
    
    # all words from unbiased sentences, except the gendered and stop words
    if args.store_data:
        store_vocab(store_dir=f'./T_DM/Data/vocab.json', texts=train_texts_19, retrieved_rem=retrieved_rem)
    vocab = retrieve_vocab(open_dir=f'./T_DM/Data/vocab.json')
    bow_vocab_size = len(vocab)

    k_values = [1,1,1,1,1]
    if args.k_values is not None:
        k_values = args.k_values

    if args.store_data:
        train_tok = tokenizer(train_texts_19, padding=False, truncation=False, return_tensors="pt")
        torch.save(train_tok, f'./T_DM/Data/bloom_tokenized_train_texts.pt')
        test_tok = tokenizer(test_texts_19, padding=False, truncation=False, return_tensors="pt")
        torch.save(test_tok, f'./T_DM/Data/bloom_tokenized_test_texts.pt')

    train_tok = torch.load(f'./T_DM/Data/bloom_tokenized_train_texts.pt')  
    test_tok = torch.load(f'./T_DM/Data/bloom_tokenized_test_texts.pt')  

    # print(Counter([len(ids) for ids in train_tok["input_ids"]]))
    # print(Counter([len(ids) for ids in test_tok["input_ids"]]))

    model = TransformerAutoencoder(tok_vocab_size=tokenizer.vocab_size, seq_len=SEQLEN, u=args.u, s=args.s).to(device)
    D1 = GenderDiscriminator(args.u).to(device)
    D2 = GenderDiscriminator(args.s).to(device)
    D3 = BoWDiscriminator(args.u, bow_vocab_size).to(device)
    D4 = BoWDiscriminator(args.s, bow_vocab_size).to(device)

    if args.what == 'train':
        
        print('TRAINING...')
        train_dataset = DMDataset(train_texts_19, train_tok, train_labels_19, vocab, retrieved_rem)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        train_dem(model, tokenizer, 'bloom', D1, D2, D3, D4, train_dataloader, args.epochs, args.lr, k_values, args.verbose, args.save_dem, device)

    if args.what == 'eval':
        
        print('EVALUATING...')
        model = TransformerAutoencoder(tok_vocab_size=tokenizer.vocab_size, seq_len=SEQLEN, u=args.u, s=args.s)
        model.load_state_dict(torch.load(args.dem_name))
        model.to(device)

        D1 = GenderDiscriminator(args.u)
        D1.load_state_dict(torch.load(args.D1_name))
        D1.to(device)

        D2 = GenderDiscriminator(args.s)
        D2.load_state_dict(torch.load(args.D2_name))
        D2.to(device)

        test_dataset = DMDataset(test_texts_19, test_tok, test_labels_19, vocab, retrieved_rem)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

        accuracy_D1, accuracy_D2 = eval_dem_trained(model, D1, D2, test_dataloader, device, args.n_visualize)
        print('Accuracy of Discriminator 1:', accuracy_D1)
        print('Accuracy of Discriminator 2:', accuracy_D2)


