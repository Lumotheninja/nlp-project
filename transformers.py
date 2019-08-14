from P1 import TrainProbabilities, START_TOK, STOP_TOK
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from enum import IntEnum

tag_to_ix = {'O': 0, 'B-positive': 1, 'B-negative': 2, 'I-positive': 3, 'B-neutral': 4, 'I-neutral': 5, 'I-negative': 6, 'B-conflict': 7, START_TOK: 8, STOP_TOK: 9}

class Const(IntEnum):
    batch = 0
    seq = 1
    feature = 2
    epochs = 10
    lr = 1e-2
    weight_decay = 1e-4
    tag_size = len(tag_to_ix)
        

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        dim_k = k.size(-1)  # size of key
        assert q.size(-1) == dim_k

        # Dot product between queries and keys for each batch and position in seq
        attn = torch.bmm(q, k.transpose(Const.seq, Const.feature))  # (batch, seq, seq)
        
        attn = attn / math.sqrt(dim_k)  # scale by dimensionality
        attn = torch.exp(attn)
        
        if mask is not None:
            attn = attn.masked_fill(mask, 0)

        attn = attn / attn.sum(dim=-1, keepdim=True)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)  # (batch, seq, feature)
        
        return output

class AttentionHead(nn.Module):
    def __init__(self, dim_model, dim_feat, dropout=0.1):
        super().__init__()
        # assume queries, keys, and values have the same feature size
        self.attn = ScaledDotProductAttention(dropout)
        self.query = nn.Linear(dim_model, dim_feat)
        self.key = nn.Linear(dim_model, dim_feat)
        self.value = nn.Linear(dim_model, dim_feat)
    
    def forward(self, queries, keys, values, mask=None):
        Q = self.query(queries)  # (batch, seq, feature)
        K = self.key(keys)  # (batch, seq, feature)
        V = self.value(values)  # (batch, seq, feature)
        x = self.attn(Q, K, V)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model, dim_feat, n_heads, dropout=0.1):
        super().__init__()
        self.dim_model = dim_model
        self.dim_feat = dim_feat
        self.n_heads = n_heads
        assert dim_model == dim_feat * n_heads

        self.attn_heads = nn.ModuleList(
            [AttentionHead(dim_model, dim_feat, dropout) for _ in range(n_heads)]
        )
        self.projection = nn.Linear(dim_feat * n_heads, dim_model)
    
    def forward(self, queries, keys, values, mask=None):
        x = [attn(queries, keys, values, mask=mask) for attn in self.attn_heads]
        x = torch.cat(x, dim=Const.feature)  # (batch, seq, dim_feat * n_heads)
        x = self.projection(x)  # (batch, seq, dim_model)
        return x

class LayerNorm(nn.Module):
    def __init__(self, dim_model, eps=1e-8):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(dim_model))
        self.beta = nn.Parameter(torch.zeros(dim_model))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class EncoderBlock(nn.Module):
    def __init__(self, dim_model=512, dim_feat=64, dim_ff=2048, n_heads=8, dropout=0.1):
        super().__init__()
        self.attn_head = MultiHeadAttention(dim_model, dim_feat, n_heads, dropout)
        self.layer_norm1 = LayerNorm(dim_model)
        self.dropout = nn.Dropout(dropout)
        self.position_wise_feed_forward = nn.Sequential(
            nn.Linear(dim_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_model),
        )
        self.layer_norm2 = LayerNorm(dim_model)
    
    def forward(self, x, mask=None):
        attn = self.attn_head(x, x, x, mask=mask)
        # Normalization and residual connection
        x = x + self.dropout(self.layer_norm1(attn))
        pos = self.position_wise_feed_forward(x)
        x = x + self.dropout(self.layer_norm2(pos))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, dim_model=512, dim_feat=64, dim_ff = 2048, n_heads=8, dropout=0.1):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(dim_model, dim_feat, n_heads, dropout)
        self.attn_head = MultiHeadAttention(dim_model, dim_feat, n_heads, dropout)
        self.position_wise_feed_forward = nn.Sequential(
            nn.Linear(dim_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_model),
        )
        self.layer_norm1 = LayerNorm(dim_model)
        self.layer_norm2 = LayerNorm(dim_model)
        self.layer_norm3 = LayerNorm(dim_model)
        self.dropout = nn.Dropout(dropout)
        
    
    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        # attention on inputs
        attn = self.masked_attn_head(x, x, x, mask=src_mask)
        x = x + self.dropout(self.layer_norm1(attn))

        attn = self.attn_head(queries=x, keys=enc_out, values=enc_out, mask=tgt_mask)
        x = x + self.dropout(self.layer_norm2(attn))
        pos = self.position_wise_feed_forward(x)
        x = x + self.dropout(self.layer_norm3(pos))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, n_blocks=6, dim_model=512, n_heads=8, dim_ff=2048, dropout=0.1):
        super().__init__()
        self.encoders = nn.ModuleList([
            EncoderBlock(dim_model=dim_model, dim_feat=dim_model // n_heads, dim_ff=dim_ff, dropout=dropout)
            for _ in range(n_blocks)
        ])

    def forward(self, x, mask=None):
        for encoder in self.encoders:
            x = encoder(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, n_blocks=6, dim_model=512, dim_feat=64, dim_ff=2048, n_heads=8, dropout=0.1, tag_size=Const.tag_size):
        super().__init__()
        self.position_emb = PositionalEmbedding(dim_model)
        self.decoders = nn.ModuleList([
            DecoderBlock(dim_model=dim_model, dim_feat=dim_model // n_heads, dim_ff=dim_ff, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.fc = nn.Linear(dim_model, tag_size)
        
    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        for decoder in self.decoders:
            x = decoder(x, enc_out, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.fc(x)

class PositionalEmbedding(nn.Module):

    def __init__(self, dim_model, max_len=512):
        super().__init__()
        pos_emb = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * -(math.log(10000.0) / dim_model))
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)
        pos_emb = pos_emb.unsqueeze(0)
        self.weight = nn.Parameter(pos_emb, requires_grad=False)
    
    def forward(self, x):
        return self.weight[:, :x.size(1), :]  # (1, seq, feat)

class WordPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, dim_model=512):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, dim_model)
        self.position_emb = PositionalEmbedding(dim_model)
    
    def forward(self, x, mask=None):
        return self.word_emb(x) + self.position_emb(x)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] if to_ix.get(w) else len(to_ix)-1 for w in seq ]
    return torch.tensor(idxs, dtype=torch.long)

def parse_training_file(path):
    data = []
    word_to_ix = {}
    with open(path, mode='r', encoding="utf-8") as f:
        running_x = []
        running_y = []
        for line in f:
            # End of a sequence
            if line=='\n':
                data.append((running_x, running_y))
                running_x = []
                running_y = []
                continue

            # Extract and format word and tag
            x,y = line.split()
            if x not in word_to_ix:
                word_to_ix[x] = len(word_to_ix)
            running_x.append(x)
            running_y.append(y)
    return data, word_to_ix

def parse_test_file(path):
    test_data = []
    with open(path, mode='r', encoding='utf-8') as f:
        running_x = []
        for line in f:
            if line == '\n':
                test_data.append(running_x)
                running_x = []
                continue
            running_x.append(line.rstrip('\n'))
    return test_data

def train(f_path):
    training_data, word_to_ix = parse_training_file(f_path)
    global vocab_len = len(word_to_ix) + 1 for "UNK"
    emb = WordPositionEmbedding(vocab_len)
    encoder = TransformerEncoder()
    decoder = TransformerDecoder()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=Const.lr, weight_decay=Const.weight_decay)
    for epoch in range(Const.epochs):
        for x, y in training_data:
            encoder.zero_grad()
            decoder.zero_grad()
            src_ids = torch.tensor([word_to_ix[word] for word in x], dtype=torch.long).unsqueeze(0)
            tgt_ids = torch.tensor([tag_to_ix[tag] for tag in y], dtype=torch.long).unsqueeze(0)
            x = encoder(emb(src_ids))
            yhat = decoder(emb(tgt_ids), x)
            loss = nn.CrossEntropyLoss()(yhat.squeeze(), tgt_ids.squeeze())
            loss.backward()
            optimizer.step()
    torch.save(emb.state_dict(), 'embeddings_ES.pt')
    torch.save(encoder.state_dict(), 'encoder_ES.pt')
    torch.save(decoder.state_dict(), 'decoder_ES.pt')

def test(f_path, out_path):
    emb = WordPositionEmbedding(vocab_len)
    encoder = TransformerEncoder()
    decoder = TransformerDecoder()

    encoder.load_state_dict(torch.load('encoder_ES.pt'))
    decoder.load_state_dict(torch.load('decoder_ES.pt'))
    embeddings.load_state_dict(torch.load('embeddings_ES.pt'))
    
    encoder.eval()
    decoder.eval()
    embeddings.eval()

    ix_to_tag = {v:k for k,v in tag_to_ix.items()}

    test_data = parse_test_file(f_path)
    output = []
    for x in test_data:
        src_ids = torch.tensor([word_to_ix[word] for word in x if word in word_to_ix else len(vocab_len)], dtype=torch.long).unsqueeze(0) #UNK word
        x = encoder(emb(src_ids))
        y = decoder(emb(tgt_ids), x)
        tags = [ix_to_tag[t] for t in y]
        output.append((sentence, tags))

    with open(out_path, mode="w", encoding='utf-8') as f:
        for sentence, tags in output:
            for word, tag in zip(sentence, tags):
                f.write(f"{word} {tag}\n")
            f.write("\n")

if __name__ == "__main__":
    train('data/ES/train')
    
