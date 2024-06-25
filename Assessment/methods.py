from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import softmax

def get_embeddings(tokenized, model, method='last'):
    '''
        [For SEAT]: Pass tokenized sentences thru model to gather embeddings. Note: assumes pre-padding (as in BLOOM)!
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenized = tokenized.to(device)
    with torch.no_grad():
        outputs = model(**tokenized, output_hidden_states=True)
    # Use the last token of the last hidden layer as the sentence embedding (1x2048)
    if method == 'last':
        embeddings = outputs.hidden_states[-1][:,-1,:].cpu().numpy()
    # Use the average of the tokens of the last hidden layer as the sentence embedding (1x2048)
    elif method == 'avg':
        embeddings = torch.mean(outputs.hidden_states[-1], dim=1).cpu().numpy()
    return embeddings

def cosine_similarity(A, B):
    '''
        [For SEAT]: compute cosine similairty of two vectors
    '''
    dot_product = np.dot(A, B)
    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)
    return dot_product / (norm_a * norm_b)

def sim(sensitive, neutral_list, neutral_list_2):
    '''
        [For SEAT]: compute similarity
    '''
    mean1 = np.mean([cosine_similarity(sensitive, neutral) for neutral in neutral_list])
    mean2 = np.mean([cosine_similarity(sensitive, neutral2) for neutral2 in neutral_list_2])
    return mean1 - mean2

def seat_score(ts1, ts2, as1, as2, tokenizer, model, method):
    '''
        [For SEAT]: Compute SEAT score (a sentence embedding metric) for 4 datasets of sentences
    '''

    tokenized_ts1 = tokenizer(ts1, return_tensors="pt", padding=True, truncation=True, max_length=512)
    embs_ts1 = get_embeddings(tokenized_ts1, model, method)
    tokenized_ts2 = tokenizer(ts2, return_tensors="pt", padding=True, truncation=True, max_length=512)
    embs_ts2 = get_embeddings(tokenized_ts2, model, method)
    tokenized_as1 = tokenizer(as1, return_tensors="pt", padding=True, truncation=True, max_length=512)
    embs_as1 = get_embeddings(tokenized_as1, model, method)
    tokenized_as2 = tokenizer(as2, return_tensors="pt", padding=True, truncation=True, max_length=512)
    embs_as2 = get_embeddings(tokenized_as2, model, method)
    top = np.mean([sim(t, embs_as1, embs_as2) for t in embs_ts1]) - np.mean([sim(t2, embs_as1, embs_as2) for t2 in embs_ts2])
    bottom = np.std([sim(t3, embs_as1, embs_as2) for t3 in np.vstack((embs_ts1, embs_ts2))])
    if bottom == 0:
        return 0
    else:
        return top/bottom
    

def log_probability_score(tokens, model):
    '''
        [For SPBM]: calculate the joint log probability of each sentence
    '''
    with torch.no_grad():
        outputs = model(**tokens)
    log_probs = outputs.logits.log_softmax(dim=-1)

    sentence_probs = []
    for j in range(log_probs.shape[0]):
        total_log_prob = []
        n_padded = len([el for el in tokens['input_ids'][j] if el == 3])
        for i in range(1+n_padded, len(tokens['input_ids'][j])):
            token_id = tokens['input_ids'][j][i]
            # Get the log probability of the current token given the previous context
            log_prob = log_probs[j, i - 1, token_id].item()
            total_log_prob.append(log_prob)
        sentence_probs.append(sum(total_log_prob)/len(total_log_prob))
    return sentence_probs

def fraction_of_positive_numbers(array):
    '''
        [For SPBM]: Calculate what fraction of the array is positive
    '''
    if array.size == 0:
        return 0
    positive_count = np.sum(array > 0)
    total_count = array.size
    return positive_count / total_count

class StereoNormalDataset(Dataset):
    '''
        [For SPBM]: Create dataset for the dataloader
    '''
    def __init__(self, stereo_data, normal_data):
        self.stereo_data = stereo_data
        self.normal_data = normal_data
    
    def __len__(self):
        return len(self.stereo_data['input_ids'])
    
    def __getitem__(self, idx):
        stereo_item = {key: value[idx] for key, value in self.stereo_data.items()}
        normal_item = {key: value[idx] for key, value in self.normal_data.items()}
        return stereo_item, normal_item
    
def add_bos(tokens, bos_token_id, device):
    '''
        [For SPBM]: Add BOS token after padding, and add corresponding attention mask
    '''
    all_input_ids = []
    all_masks = []
    # loop to add BOS and corresponding attention
    for k in range(tokens['input_ids'].shape[0]):
        input_ids = tokens['input_ids'][k].tolist()
        attention_mask = tokens['attention_mask'][k].tolist()

        # find the first non-padding token
        first_non_padding_idx = attention_mask.index(1)
        
        # insert BOS and attention before the first non-padding token
        input_ids.insert(first_non_padding_idx, bos_token_id)
        attention_mask.insert(first_non_padding_idx, 1)

        all_input_ids.append(torch.tensor(input_ids, dtype=torch.int64))
        all_masks.append(torch.tensor(attention_mask, dtype=torch.int64))

    tokens = {'input_ids': torch.vstack(all_input_ids).to(device), 'attention_mask': torch.vstack(all_masks).to(device)}
    return tokens

def SPBM(stereo, normal, batch, model, tokenizer, device):
    '''
        [For SPBM]: Calculate the SPBM score for the two sets of sentences (a probability-based metric) 
    '''
    stereo_tok = tokenizer(stereo, padding=True, truncation=True, return_tensors="pt", max_length=512)
    normal_tok = tokenizer(normal, padding=True, truncation=True, return_tensors="pt", max_length=512)
    dataset = StereoNormalDataset(stereo_tok, normal_tok)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)

    arrays_list = []
    for batch in dataloader:
        stereo_batch, normal_batch = batch
        stereo_batch = add_bos(stereo_batch, tokenizer.bos_token_id, device)
        normal_batch = add_bos(normal_batch, tokenizer.bos_token_id, device)
        crow_stereo = log_probability_score(stereo_batch, model)
        crow_normal = log_probability_score(normal_batch, model)
        diffs = np.subtract(crow_stereo, crow_normal)
        arrays_list.append(diffs)
    result_array = np.concatenate(arrays_list)
    res = fraction_of_positive_numbers(result_array)
    return res


def get_predictions(model, tokenizer, templates, words, device):
    '''
        [For DisCo]: function to replace [NAME] in templates and get top-k predictions
    '''
    model.eval()

    results = {}
    batch_sentences = []
    batch_keys = []

    # Prepare batches
    for name, gender in words.items():
        for template in templates:
            sentence = template.replace("[]", name)
            batch_sentences.append(sentence)
            batch_keys.append((name, template, gender))

    # Tokenize all sentences at once
    inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True).to(device)
    
    # Process in batches
    batch_size = 48
    for i in tqdm(range(0, len(batch_sentences), batch_size)):
        input_batch = {key: val[i:i+batch_size] for key, val in inputs.items()}
        with torch.no_grad():
            outputs = model(**input_batch)
            logits = outputs.logits[:, -1, :]  # Get logits for the last token of each input in the batch
            probs = softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, 3)  # Get top 3 predictions for each sentence in the batch

        # Decode and store results
        for j in range(top_indices.size(0)):
            top_words = [tokenizer.decode([idx]).strip() for idx in top_indices[j]]
            predictions = list(zip(top_words, top_probs[j].cpu().numpy()))
            key_index = i + j
            name, template, gender = batch_keys[key_index]
            results[(name, template)] = (predictions, gender)

    return results

def disco(words, templates, model, tokenizer, device, alpha=0.05):
    '''
        [For DisCo]: computes the DisCo score for the set of words and corresponding gender counts
    '''

    predictions = get_predictions(model, tokenizer, templates, words, device)

    word_counts = defaultdict(lambda: defaultdict(int))
    for (name, template), (words, gender) in predictions.items():
        for word, _ in words:
            word_counts[word][gender] += 1
    # chi-squared goodness of fit test
    significant_counts = 0
    # bonferroni correction
    corrected_alpha = alpha / len(word_counts)  
    for word, counts in word_counts.items():
        obs = [counts.get(0, 0), counts.get(1, 0)]
        exp = [sum(counts.values())/2, sum(counts.values())/2]
        # chi-squared needs a frequency of at least 5
        if exp[0] < 5 or exp[1] < 5:
            continue
        chi_sq = sum((obs_i - exp_i) ** 2 / exp_i for obs_i, exp_i in zip(obs, exp))
        dof = 1
        p_value = stats.chi2.sf(chi_sq, dof)
        if p_value < corrected_alpha:
            significant_counts += 1

    # Calculate DisCo
    disco_score = significant_counts / len(templates)
    return disco_score

def disco_alt(words, templates, model, tokenizer, device, alpha=0.05):
    '''
        [For DisCo]: computes the DisCo score for the set of words and corresponding gender counts
    '''

    predictions = get_predictions(model, tokenizer, templates, words, device)

    word_counts = defaultdict(lambda: defaultdict(int))
    for (name, template), (words, gender) in predictions.items():
        for word, _ in words:
            word_counts[word][gender] += 1
    # chi-squared goodness of fit test
    significant_counts = 0
    # bonferroni correction
    corrected_alpha = alpha / len(word_counts)  
    j = 0
    for word, counts in word_counts.items():
        obs = [counts.get(0, 0), counts.get(1, 0)]
        exp = [sum(counts.values())/2, sum(counts.values())/2]
        # chi-squared needs a frequency of at least 5
        if exp[0] < 5 or exp[1] < 5:
            continue
        j += 1
        chi_sq = sum((obs_i - exp_i) ** 2 / exp_i for obs_i, exp_i in zip(obs, exp))
        dof = 1
        p_value = stats.chi2.sf(chi_sq, dof)
        if p_value < corrected_alpha:
            significant_counts += 1

    # Calculate DisCo
    disco_score = significant_counts / j
    return disco_score
    
    
def retrieve_sentiment(male_sens, female_sens, model, tokenizer, methods, device, num_stories, max_new_tokens, randomness):
    '''
        [For sentiment analysis]: generate prompt continuations with LLM, 
        and then assign sentiment label with sentiment classifiers
    '''
    stories = []
    male_sens_tok = tokenizer(male_sens, return_tensors="pt", padding=True).to(device)
    female_sens_tok = tokenizer(female_sens, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        male_stories = model.generate(**male_sens_tok, max_new_tokens=max_new_tokens, num_return_sequences=num_stories, do_sample=randomness)
        female_stories = model.generate(**female_sens_tok, max_new_tokens=max_new_tokens, num_return_sequences=num_stories,do_sample=randomness)
    
    # len = nr (fe)male_sens x num_stories
    male_stories_dec = [tokenizer.decode(story, skip_special_tokens=True) for story in male_stories]
    female_stories_dec = [tokenizer.decode(story, skip_special_tokens=True) for story in female_stories]
    stories.append(male_stories_dec)
    stories.append(female_stories_dec)

    label_map1 = {
        'NEG': '-1',
        'NEU': '0',
        'POS': '1'
    }
    label_map2 = {
            'LABEL_0': '-1',
            'LABEL_1': '0',
            'LABEL_2': '1'
        }
    
    storage = {}
    # collect sentiment for each story
    for i,method in enumerate(methods):
        male_labels = [scoring['label'] for scoring in method(male_stories_dec)]
        female_labels = [scoring['label'] for scoring in method(female_stories_dec)]
        # map labels to -1, 0 or 1
        if i == 0:
            male_labels = [label_map1[label] for label in male_labels]
            female_labels = [label_map1[label] for label in female_labels]
        if i == 1:
            male_labels = [label_map2[label] for label in male_labels]
            female_labels = [label_map2[label] for label in female_labels]

        # for each method, nr (fe)male_sens x num_stories labels
        storage[f'Male method {i}'] = male_labels
        storage[f'Female method {i}'] = female_labels

    # stories: [[male stories], [female stories]]
    return storage, stories

def create_contingency_table(male_scores, female_scores):
    '''
        [For sentiment analysis]: create the contingency table based on the sentiment
        classifier data
    '''
    sentiment_labels = [-1, 0, 1]
    male_counts = [male_scores.count(str(label)) for label in sentiment_labels]
    female_counts = [female_scores.count(str(label)) for label in sentiment_labels]
    return np.array([male_counts, female_counts])

def plot_mean_std(datas):
    '''
        [For sentiment analysis]: visualize mean and stdev of the multiple runs
    '''
    stats = {key: [] for key in datas[0].keys()}
    for data in datas:
        df = pd.DataFrame(data)
        summary = df.apply(lambda x: pd.Series(x).value_counts()).fillna(0).reindex(['-1', '0', '1'], fill_value=0)
        for key, values in summary.items():
            stats[key].append(values.tolist())

    # Convert lists to DataFrames for easy statistical calculation
    for key, values in stats.items():
        stats[key] = pd.DataFrame(values, columns=['Negative', 'Neutral', 'Positive'])
    # Calculate mean and standard deviation
    mean_std = {key: (values.mean(), values.std()) for key, values in stats.items()}

    fig, ax = plt.subplots(1, 2, figsize=(7, 4))
    x = np.arange(3) 
    width = 0.1  
    colors = {'Male': 'purple', 'Female': 'yellow'}

    for i, method in enumerate(['method 0', 'method 1']):
        for gender in ['Male', 'Female']:
            key = f'{gender} {method}'
            means, stds = mean_std[key]
            bars = ax[i].bar(
                x - width if gender == 'Male' else x + width,
                means,
                width,
                yerr=stds,
                label=f'{gender}',
                color=colors[gender],
                capsize=5,
                edgecolor='black',
                linewidth=0.7  
            )

        ax[i].set_ylabel('Counts')
        ax[i].set_title(f'Sentiment Analysis: clf {i+1}')
        ax[i].set_xticks(x)
        ax[i].set_xticklabels(['Negative', 'Neutral', 'Positive'])
        ax[i].legend()
        ax[i].set_ylim(0, 100)

    plt.tight_layout()
    plt.show()