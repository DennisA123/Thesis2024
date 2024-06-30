from dem import *
from discriminators import *
from utils import *
import torch
import argparse
import sys
from torch.utils.data import DataLoader
import torch.optim as optim
from collections import Counter
import json
from tqdm import tqdm
import torch.nn.functional as F
from transformers import BloomForCausalLM, BloomTokenizerFast
import random
from sklearn.model_selection import train_test_split

# python ./Debiasing/train_dialogue.py --save_G
parser = argparse.ArgumentParser()
parser.add_argument('--store_data', action='store_true', help='Create the necessary files if they do not already exist')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--prompts_len', type=int, default=20, help='Max length for output of LM')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizers')
parser.add_argument('--loop', type=int, default=50, help='How often to do the training loop')
parser.add_argument('--u', type=int, default=500, help='unbiased gender vector size')
parser.add_argument('--s', type=int, default=500, help='semantic vector size')
parser.add_argument('--save_G', action='store_true', help='Store dialogue model')
parser.add_argument('--dem_name', type=str, default='./Models/gru_dm.pt')

args = parser.parse_args()

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_name = "bigscience/bloom-560m"
    G_model = BloomForCausalLM.from_pretrained(model_name).to(device)

    # only train certain layers
    N = 10
    for param in G_model.parameters():
        param.requires_grad = False
    num_layers = len(G_model.transformer.h)
    print('Number of layers in LLM:', num_layers)
    layers_to_finetune = [G_model.transformer.h[i] for i in range(num_layers - N, num_layers)]
    for layer in layers_to_finetune:
        for param in layer.parameters():
            param.requires_grad = True

    tokenizer = BloomTokenizerFast.from_pretrained(model_name)

    THEIR_dem_opt = {'emb_size': 300, 'hidden_size': 1000, 'unbias_size': 500, 'content_size': 500, 'rnn_class': nn.GRU, 'dropout': 0.0}
    THEIR_dem = Autoencoder(emb_size=THEIR_dem_opt['emb_size'], hidden_size=THEIR_dem_opt['hidden_size'], unbias_size=THEIR_dem_opt['unbias_size'], content_size=THEIR_dem_opt['content_size'],
                    dict_file='./GRU_DM/dict.dict',
                    dropout=THEIR_dem_opt['dropout'], rnn_class=THEIR_dem_opt['rnn_class'], device=device).to(device)
    THEIR_dem.load_state_dict(torch.load(args.dem_name))

    D1 = GenderDiscriminator(input_dim=args.u).to(device)
    D2 = GenderDiscriminator(input_dim=args.s).to(device)

    cross_ent_criterion = nn.CrossEntropyLoss()

    with open('./Debiasing/Data/adv_train_data.json', 'r') as f:
        train_gender_data_list, n_data = json.load(f)

    random.seed(8)
    g_label_0 = [datapoint for datapoint in train_gender_data_list if datapoint[2] == 0]
    g_label_1 = [datapoint for datapoint in train_gender_data_list if datapoint[2] == 1]
    # shuffle such that random data is selected when only taking first 60 000
    random.shuffle(g_label_0)
    random.shuffle(g_label_1)
    g_balanced_label_0 = g_label_0[:60000]
    g_balanced_label_1 = g_label_1[:60000]
    g_data = g_balanced_label_0 + g_balanced_label_1
    g_train, g_both = train_test_split(g_data, test_size=0.2, random_state=8) # important: same random seed as in evaluation!
    _, _ = train_test_split(g_both, test_size=0.5, random_state=8)

    n_train, n_both = train_test_split(n_data, test_size=0.2, random_state=8)
    _, _ = train_test_split(n_both, test_size=0.5, random_state=8)

    X_gender, Y_gender, g = load_data_gender(g_train)
    X_neutral, Y_neutral = load_data_neutral(n_train)

    XY_gender = combine_lists(X_gender, Y_gender, separator=' ')
    XY_neutral = combine_lists(X_neutral, Y_neutral, separator=' ')

    if args.store_data:
        print('Storing the data...')
        encoded_X_gender = tokenizer(X_gender, padding=True, truncation=True, return_tensors="pt", max_length=64)
        torch.save(encoded_X_gender, './Debiasing/Data/tokenized_X_gender_G.pt')
        encoded_Y_gender = tokenizer(Y_gender, padding=True, truncation=True, return_tensors="pt", max_length=1)
        torch.save(encoded_Y_gender, './Debiasing/Data/tokenized_Y_gender_G.pt')
        encoded_XY_gender = tokenizer(XY_gender, padding=True, truncation=True, return_tensors="pt", max_length=96)
        torch.save(encoded_XY_gender, './Debiasing/Data/tokenized_XY_gender_G.pt')
        encoded_XY_neutral = tokenizer(XY_neutral, padding=True, truncation=True, return_tensors="pt", max_length=64)
        torch.save(encoded_XY_neutral, './Debiasing/Data/tokenized_XY_neutral_G.pt')
        print('Stored the data in the directory!')

    print('Loading the data...')
    loaded_encoded_X_gender = torch.load('./Debiasing/Data/tokenized_X_gender_G.pt')
    loaded_encoded_Y_gender = torch.load('./Debiasing/Data/tokenized_Y_gender_G.pt')
    loaded_encoded_XY_gender = torch.load('./Debiasing/Data/tokenized_XY_gender_G.pt')
    loaded_encoded_XY_neutral = torch.load('./Debiasing/Data/tokenized_XY_neutral_G.pt')
    print('Loaded the data!')

    D_g = TripletDataset(loaded_encoded_X_gender, loaded_encoded_Y_gender, g)
    D_g_combined = CombinedDataset(loaded_encoded_XY_gender)
    D_n_combined = CombinedDataset(loaded_encoded_XY_neutral)
    D_g_loader = DataLoader(D_g, batch_size=args.batch_size, shuffle=False) # NOTE: these need to match! so not shuffled
    D_g_combined_loader = DataLoader(D_g_combined, batch_size=args.batch_size, shuffle=False)
    D_n_combined_loader = DataLoader(D_n_combined, batch_size=args.batch_size, shuffle=True)

    D2_optimizer = optim.Adam(D2.parameters(), lr=args.lr)
    G_TRAINABLE_PARAMS = [param for param in G_model.parameters() if param.requires_grad]
    G_D1_optimizer = optim.Adam(G_TRAINABLE_PARAMS+list(D1.parameters()), lr=args.lr)
    G_optimizer = optim.Adam(G_TRAINABLE_PARAMS, lr=args.lr)
    # ce_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    for q, (dg_batch, dg_comb_batch, dn_comb_batch) in tqdm(enumerate(zip(D_g_loader, D_g_combined_loader, D_n_combined_loader)), total=min(len(D_g_loader), len(D_n_combined_loader))):
        
        # train D2 to make correct predictions ------------------------------------------
        D2.train()
        G_model.eval()

        x, _, g = dg_batch
        x_tokenized = x['input_ids'].to(device)
        x_attention_mask = x['attention_mask'].to(device)
        g = g.to(device)

        G_output = generate_response(x_tokenized, args.prompts_len, x_attention_mask, G_model)
        G_output_trimmed = G_output[:, -args.prompts_len:]
        G_output_trimmed_their, g = their_tokenizer([tokenizer.decode(ids, skip_special_tokens=True) for ids in G_output_trimmed], g, THEIR_dem, args.batch_size, THEIR_dem.dict.tok2ind[THEIR_dem.dict.unk_token], THEIR_dem.dict.tok2ind[THEIR_dem.dict.null_token])
        G_output_trimmed_their = G_output_trimmed_their.to(device)
        
        _, fs = THEIR_dem.forward_encoder_preclass(G_output_trimmed_their)

        ps = D2(fs)
        ce_loss_D2 = cross_ent_criterion(ps, g)
        print('LOSS 1:', ce_loss_D2)
        D2_optimizer.zero_grad()
        ce_loss_D2.backward(retain_graph=True)
        D2_optimizer.step()

        # train LLM and D1 --------------------------------------------------------------
        D1.train()
        D2.eval()
        G_model.train()

        x, _, g = dg_batch
        x_tokenized = x['input_ids'].to(device)
        x_attention_mask = x['attention_mask'].to(device)
        g = g.to(device)

        G_output = generate_response(x_tokenized, args.prompts_len, x_attention_mask, G_model)
        G_output_trimmed = G_output[:, -args.prompts_len:]
        G_output_trimmed_their, g = their_tokenizer([tokenizer.decode(ids, skip_special_tokens=True) for ids in G_output_trimmed], g, THEIR_dem, args.batch_size, THEIR_dem.dict.tok2ind[THEIR_dem.dict.unk_token], THEIR_dem.dict.tok2ind[THEIR_dem.dict.null_token])
        G_output_trimmed_their = G_output_trimmed_their.to(device)

        fu, fs = THEIR_dem.forward_encoder_preclass(G_output_trimmed_their)

        pu = D1(fu)
        ps = F.softmax(D2(fs), dim=1)

        loss_D1 = cross_ent_criterion(pu, g)
        loss_D2 = -entropy(ps, ps)

        xy = dg_comb_batch
        xy_tokenized = xy['input_ids'].to(device)
        xy_attention_mask = xy['attention_mask'].to(device)
        xy_labels = xy['input_ids'].to(device)
        xy_labels[xy_labels == tokenizer.pad_token_id] = -100

        outputs = G_model(input_ids=xy_tokenized, attention_mask=xy_attention_mask, labels=xy_labels)
        mle_loss = outputs.loss

        loss = mle_loss + loss_D1 + loss_D2
        print('LOSS 2:', loss)
        G_D1_optimizer.zero_grad()
        loss.backward()
        G_D1_optimizer.step()
        
        # train LLM on neutral dataset ---------------------------------------------------------

        G_model.train()

        xy = dn_comb_batch
        xy_tokenized = xy['input_ids'].to(device)
        xy_attention_mask = xy['attention_mask'].to(device)
        xy_labels = xy['input_ids'].to(device)
        xy_labels[xy_labels == tokenizer.pad_token_id] = -100

        outputs = G_model(input_ids=xy_tokenized, attention_mask=xy_attention_mask, labels=xy_labels)
        mle_loss_2 = outputs.loss
        print('LOSS 3:', mle_loss_2)

        G_optimizer.zero_grad()
        mle_loss_2.backward()
        G_optimizer.step()

        if args.save_G and q % 20 == 0:
            print(f'Saving model checkpoint (iteration {q})')
            G_model.save_pretrained(f'./Models/DB_BLOOM')

        # if q == args.loop:
        #     break