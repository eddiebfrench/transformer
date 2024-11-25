import torch
import os
from datasets import load_dataset
import re
import random
import numpy as np
from unidecode import unidecode
import sentencepiece as spm
import torch.nn as nn
from torch.nn import functional as F

block_size = 256
n_embd = 64*4
batch_size = 32
vocab_size = 10000
dropout = 0.2
device = 'cuda'
sp = spm.SentencePieceProcessor(model_file='bpe.model')
encode = lambda s: sp.encode(s, out_type=int)
decode = lambda s: sp.decode(s)


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential( #this makes things very deep. This is why we need residual connections. Addition distributes gradients equally to its two branches
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            nn.LayerNorm(n_embd),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape

        tok_emb = self.token_embedding_table(idx) #(Batch by Time by Channel)
        pos_emb = self.position_embedding_table(torch.arange(T,  device=device)) #T by C
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x) # (B by T by vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets, ignore_index=9999)

        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=0.5, last_k=10, penalty=1.5, training=True):
        #idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            #crop idx to last block_size tokens
            idx_cond = idx[:, -block_size:] #(B, C)
            #get the predictions
            logits, loss = self(idx_cond)
            #focus only on the last time step
            logits = logits[:, -1, :] #becomes (B,C)
            for b in range(idx.shape[0]):  # Iterate over the batch
                recent_tokens = idx[b, -last_k:] if idx.shape[1] > last_k else idx[b]
                for token in recent_tokens:
                    logits[b, token] /= penalty
            #apply softmax to get probabilities
            probs = F.softmax(logits/temperature, dim=-1) #(B,C)
            #sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B, 1)
            #append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) #(B, T+1)
            if training==False and (decode(idx[0].tolist()[-2:]) in ["A:", "B:"] or idx_next==9999):
                break
        return idx
class Head(nn.Module):
    """ one head of self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size, device=device)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        
        wei = q @ k.transpose(-2, -1) * C**-0.5 #(B,T,C) @ (B, C, T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        #perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v #(B,T,T) @ (B,T,C) -> (B,T,C)

        return out
class MultiHeadAttention(nn.Module):
    """multiple heads of self attention"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) #linear transformation of the output 
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, C) concatenating over the channel dimension
        out = self.dropout(self.proj(out))
        return out 

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """transformer block: communication followed by computation"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)


    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x