import torch
from torch.utils.data import Dataset

def pad(list, padding=0, min_len=None):
        padded = []
        max_len = max([len(l) for l in list])
        if min_len:
            max_len = max(min_len, max_len)
        for l in list:
            padded.append(l + [padding] * (max_len - len(l)))

        return torch.tensor(padded, dtype=torch.long)

def entropy(x,y):
    '''
        Calculates the entropy
    '''
    return -torch.sum(x * torch.log(y + 1e-9), dim=1).mean()

def load_data_gender(data):
    '''
        Retrieve texts, responses and genders
    '''
    x = [item[0] for item in data]
    y = [item[1] for item in data]
    g = [item[2] for item in data]
    return x, y, g

def load_data_neutral(data):
    '''
        Retrieve texts and responses
    '''
    x = [item[0] for item in data]
    y = [item[1] for item in data]
    return x, y

def combine_lists(list1, list2, separator=' '):
    '''
        Combine elements of two lists into one, with a separator string in between each pair
    '''
    return [f'{a}{separator}{b}' for a, b in zip(list1, list2)]

class CombinedDataset(Dataset):
    '''
        Create dataset of concats of text-response pairs
    '''
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts

    def __len__(self):
        return len(self.tokenized_texts['input_ids'])

    def __getitem__(self, idx):
        text = {key: val[idx] for key, val in self.tokenized_texts.items()}
        return text

class TripletDataset(Dataset):
    '''
        Create the gendered dataset for the debiasing loop
    '''
    def __init__(self, tokenized_texts, tokenized_responses, genders):
        self.tokenized_texts = tokenized_texts
        self.tokenized_responses = tokenized_responses
        self.genders = genders

    def __len__(self):
        return len(self.tokenized_texts['input_ids'])

    def __getitem__(self, idx):
        text = {key: val[idx] for key, val in self.tokenized_texts.items()}
        response = {key: val[idx] for key, val in self.tokenized_responses.items()}
        gender = self.genders[idx]
        return text, response, gender
        
def generate_response(tokenized_prompt, max_generation, attention_mask, model):
    '''
        Given a tokenized input prompt, generate the output tokens of the LLM (tokenized)
    '''
    output = model.generate(tokenized_prompt, attention_mask=attention_mask, max_new_tokens=max_generation, repetition_penalty=1.2)
    return output

def their_tokenizer(data, labels, model, bs, unk_id, null_id):
    '''
        Given LLM output, tokenize it correctly to pass to the GRU-DM
    '''
    n = 10
    G_output_tok = []
    for i in range(len(data) // (n * bs) + 1):
        nbatch = data[i * (n * bs) : (i+1) * (n * bs)]
        nbatch_list = [[model.dict.tok2ind.get(word, unk_id) for word in ins.split()] for ins in nbatch]
        descend_nbatch_list = sorted(nbatch_list, key=lambda x: len(x), reverse=True)

        j = 0
        while len(descend_nbatch_list[j * bs : (j+1) * bs]) > 0:
            batch_list = descend_nbatch_list[j * bs : (j+1) * bs]

            # text: (batch_size x seq_len)
            text = pad([x for x in batch_list], padding=null_id)
            G_output_tok.append(text)
            j += 1

    mask = torch.any(G_output_tok[0] != null_id, dim=1)
    data_filtered = G_output_tok[0][mask]
    labels = labels[mask]
    
    return data_filtered, labels