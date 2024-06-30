from utils import *
from dem import *
from eval_dem import eval_dem_trained
from sklearn.model_selection import train_test_split
import random

# nltk.download('punkt')

import torch
torch.cuda.empty_cache()
import argparse
import sys

# python ./GRU_DM/main_dem.py --dem_name ./Models/gru_dm.pt
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--dem_name', type=str, default='./Models/gru_dm.pt')
parser.add_argument('--n_visualize', type=int, default=250, help='Number of datapoints per gender to visualize with t-SNE')

args = parser.parse_args()

if __name__ == "__main__":

    random.seed(8)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dem_opt = {'emb_size': 300, 'hidden_size': 1000, 'unbias_size': 500, 'content_size': 500, 'rnn_class': nn.GRU, 'dropout': 0.0}
    model = Autoencoder(emb_size=dem_opt['emb_size'], hidden_size=dem_opt['hidden_size'], unbias_size=dem_opt['unbias_size'], content_size=dem_opt['content_size'],
                    dict_file='./GRU_DM/dict.dict',
                    dropout=dem_opt['dropout'], rnn_class=dem_opt['rnn_class'], device=device).to(device)
    print(args.dem_name)
    model.load_state_dict(torch.load(args.dem_name))

    # 0
    null_id = model.dict.tok2ind[model.dict.null_token]
    # 2
    eos_id = model.dict.tok2ind[model.dict.end_token]
    # 3
    unk_id = model.dict.tok2ind[model.dict.unk_token]
    # 30 000
    vocab_size = len(model.dict)

    with open(f'./GRU_DM/Data/unbiased_corpus.json', 'r') as f:
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
        _, both = train_test_split(balanced_data_list, test_size=0.2, random_state=8) # important: same random seed as in training!
        _, test_data_list = train_test_split(both, test_size=0.5, random_state=8)

    n = 10
    test_data = []
    for i in range(len(test_data_list) // (n * args.batch_size) + 1):
        nbatch = test_data_list[i * (n * args.batch_size) : (i+1) * (n * args.batch_size)]
        nbatch_list = [([model.dict.tok2ind.get(word, unk_id) for word in ins[0].split()],
                        ins[1]) for ins in nbatch]
        descend_nbatch_list = sorted(nbatch_list, key=lambda x: len(x[0]), reverse=True)

        j = 0
        while len(descend_nbatch_list[j * args.batch_size : (j+1) * args.batch_size]) > 0:
            batch_list = descend_nbatch_list[j * args.batch_size : (j+1) * args.batch_size]

            # text: (batch_size x seq_len)
            text = pad([x[0] for x in batch_list], padding=null_id)
            labels = torch.tensor([x[1] for x in batch_list], dtype=torch.long)
            test_data.append((text, labels))
            j += 1

    print('Test batches:', len(test_data))
    
    accuracy_D1, accuracy_D2 = eval_dem_trained(model, test_data, device, args.n_visualize)
    print('Accuracy of classifier 1:', accuracy_D1)
    print('Accuracy of classifier 2:', accuracy_D2)


