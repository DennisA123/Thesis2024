import math
import torch
import torch.nn as nn

class BoWDiscriminator(nn.Module):
    # def __init__(self, input_dim, vocab_size, hidden_dim1=1024, hidden_dim2=2048, hidden_dim3=4096):
    #     super().__init__()
    #     self.fc1 = nn.Linear(input_dim, hidden_dim1)
    #     self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
    #     self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)and
    #     self.fc4 = nn.Linear(hidden_dim3, vocab_size)
    #     self.relu = nn.ReLU()
    
    # def forward(self, x):
    #     x = self.relu(self.fc1(x))
    #     x = self.relu(self.fc2(x))
    #     x = self.relu(self.fc3(x))
    #     return self.fc4(x)
    def __init__(self, input_dim, vocab_size):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, vocab_size)
    
    def forward(self, x):
        return self.fc1(x)

class GenderDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 2)
    
    def forward(self, x):
        return self.fc1(x)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class TransformerAutoencoder(nn.Module):
    def __init__(self, tok_vocab_size, seq_len, d_model=1000, nhead=4, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=1024, u=500, s=500):
        super(TransformerAutoencoder, self).__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.pos_encoder = PositionalEncoding(d_model, 0.5)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Projection layers
        self.to_input = nn.Embedding(tok_vocab_size, d_model)
        self.ae_encode = nn.Linear(self.seq_len*self.d_model, self.d_model)
        self.to_fx = nn.Linear(d_model, u)
        self.to_fy = nn.Linear(d_model, s)
        self.ae_decode = nn.Linear(self.d_model, self.seq_len*self.d_model)
        self.to_output = nn.Linear(d_model, tok_vocab_size)


    # input sequence: (batch size, largest seq len)
    def forward(self, input_seq, attention_mask):
        if attention_mask is not None:
            transformer_mask = ~attention_mask.bool()
        else:
            transformer_mask = None
            print('Warning, no attention mask provided!')

        # (batch size, largest seq len, d_model), every token becomes a vector
        encoder_input = self.to_input(input_seq) * math.sqrt(self.d_model)
        # encoder_input = self.pos_encoder(encoder_input)
        # (batch size, largest seq len, d_model)
        encoder_output = self.encoder(encoder_input, src_key_padding_mask=transformer_mask)
        # (batch size, largest seq len x d_model)
        ae_input = encoder_output.view(encoder_output.shape[0], encoder_output.shape[1] * encoder_output.shape[2])

        # (batch_size, d_model)
        h = self.ae_encode(ae_input)
        # (batch size, u)
        fu = self.to_fx(h)
        # (batch size, s)
        fs = self.to_fy(h)
        # (batch size, (u+s))
        f = torch.cat((fu, fs), dim=1)

        # (batch size, largest seq len x d_model)
        ae_output = self.ae_decode(f)
        # (batch size, largest seq len, d_model)
        decoder_input = ae_output.view(encoder_output.shape[0], encoder_output.shape[1], encoder_output.shape[2])

        # (batch size, largest seq len, d_model)
        decoder_output = self.decoder(decoder_input, encoder_output, memory_key_padding_mask=transformer_mask)
        # (batch size, largest seq len, vocab_size)
        output = self.to_output(decoder_output)

        return output, fu, fs
    
