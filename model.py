import os
import math

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

VOCAB_SIZE = 3000

BATCH_SIZE = 32
CONTEXT_WINDOW = 128
D_MODEL = 256

NUM_HEADS = 4
FFN_HIDDEN = 512
NUM_BLOCKS = 3

#This is the embedding layer and what actually gets fed into the model
#This is the simpler version for now with just regular plain embeddings
#   but I will soon add the RoPE math and everything
class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, batch):
        return self.token_embedding(batch)
    
def RoPE(d_model, num_heads, context_window):
    mat_dim = int(d_model/num_heads)
    base = 10000
    rotation_matrix = torch.zeros(context_window, mat_dim, mat_dim)

    thetas = 1 / torch.pow(base, (2 * torch.arange(0, mat_dim/2, 1) / mat_dim))
    theta_m = torch.outer(torch.arange(0, context_window, 1), thetas)

    for m in range(0, context_window):
        for i in range(0, int(mat_dim/2)):
            cos_m = torch.cos(theta_m[m,i])
            sin_m = torch.sin(theta_m[m,i])

            rotation_matrix[m, 2*i, 2*i] = cos_m
            rotation_matrix[m, 2*i, 2*i+1] = -sin_m
            rotation_matrix[m, 2*i+1, 2*i] = sin_m
            rotation_matrix[m, 2*i+1, 2*i+1] = cos_m
    
    return rotation_matrix

#Creating the actual Transfomer Architecture now
#We will do a single hidden layer for now
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.GeLU = nn.GELU()
    
    def forward(self, batch):
        return self.w2(self.GeLU(self.w1(batch)))
    
#Creating a single attention head right now
#Not the best practice but we will adjust/update later on
#Shape: (batch_size, context_window, d_model) ---> (batch_size, context_window, d_model/num_heads)
class AttentionHead(nn.Module):
    def __init__(self, d_model, num_heads, context_window):
        super().__init__()
        #Projection dim for emb_dim --> qkv dim
        self.qkv_dim = int(d_model/num_heads)

        #Intialize the QKV Matrices
        self.query = nn.Linear(d_model, self.qkv_dim, bias=False)
        self.key = nn.Linear(d_model, self.qkv_dim, bias=False)
        self.value = nn.Linear(d_model, self.qkv_dim, bias=False)

        #Creating the RoPE Matrix (not the most efficient way as of now)
        rotation_mat = RoPE(d_model, num_heads, context_window)
        self.register_buffer("rotation_mat", rotation_mat)
        
        #Creating the mask for the attention head, we are storing as attribute
        #Finds the upper triangle and sets it to ones --> bools
        mask = torch.triu(torch.ones(context_window, context_window), diagonal=1).bool()
        self.register_buffer("mask", mask)
    
    def forward(self, batch):
        #Actual Q @ K.T in attention, the transpose just swaps the last two dims in the shape
        #RoPE has Shape: (context_window, qkv_dim, qkv_dim)
        #Shape: (batch_size, context_window, qkv_dim)

        query = self.rotation_mat.unsqueeze(0) @ self.query(batch).unsqueeze(-1)
        key = self.rotation_mat.unsqueeze(0) @ self.key(batch).unsqueeze(-1)

        qk = query.squeeze(-1) @ key.squeeze(-1).mT
        attention_score_logits = qk / math.sqrt(self.qkv_dim)
        
        #Actual attention score probabilites
        #Shape: (batch_size, context_window, context_window)
        attention_score_logits.masked_fill_(self.mask, value=float("-inf"))
        attention_scores = attention_score_logits.softmax(dim=-1)

        #Shape: (batch_size, context_window, qkv_dim)
        attention_embeddings = attention_scores @ self.value(batch)

        return attention_embeddings
    
#Class for all the attention heads
#Shape: list of length num_heads (batch_size, context_window, d_model/num_heads) ---> (batch_size, context_window, d_model)
class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, num_heads, context_window):
        super().__init__()
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([
            AttentionHead(d_model, num_heads, context_window)
            for i in range(num_heads)
        ])

        #Mixing all the information from the attention heads
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, batch):
        #A list of the outputs of each attention head
        #Shape: (batch_size, context_window, d_model/num_heads)
        forward_head = [head(batch) for head in self.attention_heads]

        #Shape: (batch_size, context_window, d_model)
        output = torch.cat(forward_head, dim=-1)
        return self.out_proj(output)

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, context_window, d_ff):
        super().__init__()
        self.num_heads = num_heads

        self.multi_headed_attention = MultiHeadedAttention(
            d_model,
            num_heads,
            context_window
        )
        
        self.ffn = FeedForward(
            d_model, 
            d_ff
        )

        self.pre_RMSNorm = nn.RMSNorm(d_model)
        self.post_RMSNorm = nn.RMSNorm(d_model)

    def forward(self, X):
        #Shape: (batch_size, context_window, d_model)
        #Normalize X (Pre RMSNorm) ---> Attention ---> X + Normalize Attention Ouput (Post RMSNorm)
        X_norm = self.pre_RMSNorm(X)
        mha = self.multi_headed_attention(X_norm)

        #Make sure we don't forget the original embeddings (pre-attention)
        X_mid = X + mha
        
        #Normalize Attention Output ---> FFN ---> Final Decoder Block Output
        ffn_in = self.post_RMSNorm(X_mid)
        ffn = self.ffn(ffn_in)

        #Make sure we don't lose information after FFN
        output = X_mid + ffn
        return output

#Full walk through of Class
#   Pre-class we tokenize all data and set up the dataloaders
#       - Shape we are feeding into embedding layer |   Shape: (batch_size, context_window)
#   Embedding Layer (Pre-attention, non-context rich embeddings)    |   Shape: (batch_size, context_window, d_model)
#   3 Decoder Blocks
#   Apply RMSNorm one last time
#   LM Head to go from (batch_size, context_window, d_model) ---> (batch_size, context_window, token_count)
#       - This is our predicitions for each
#   Softmax layer and get our predicitions
class Transformer(nn.Module):
    def __init__(self, num_blocks, d_model, vocab_size, num_heads, d_ff, context_window):
        super().__init__()
        self.embedding_layer = EmbeddingLayer(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, context_window, d_ff)
            for _ in range(num_blocks)
        ])

        self.RMSNorm = nn.RMSNorm(d_model)
        self.LM_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, batch):
        X = self.embedding_layer(batch)

        for block in self.blocks:
            X = block(X)
        
        X = self.RMSNorm(X)
        logits = self.LM_head(X)
        return logits