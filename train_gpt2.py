from dataclasses import dataclass
from transformers import GPT2LMHeadModel
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)

    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class CausalSelfAttention(nn.Module):    #Attention in LLM is generally what we ussually give importance to certain things
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        # self.attn_dropout = nn.Dropout(config.attn_pdrop)
        # self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        #multi heads increase the model’s expressive capacity during training, not doing extra training.
        #Splits embedding space into multiple learned subspaces so the model can learn different interaction patterns in parallel.
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim -> query - what?, key - indexed token, value - is the token that is present
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) #here the masking is done so that the model cannot the see the future tokens, if so then it will be cheating and the model would not think of itself.
        att = F.softmax(att, dim=-1) #this generally makes the probability of the raw scores, hence the probability distribution influences how each token influences the current token.
        # att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs) --> then applying attention to values
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        # y = self.resid_dropout(self.c_proj(y))
        y = self.c_proj(y)
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)  # normally a aggregation function/weighted sum/reducing
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # self.mlp = nn.ModuleDict(dict(
        #     c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
        #     c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
        #     act     = NewGELU(),
        #     dropout = nn.Dropout(config.resid_pdrop),
        # ))
        # m = self.mlp
        # self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward
        self.mlp = MLP(config)
#attention is like reduce and mlp does like mapping --> so its similar like map reduce
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  ##These are the layers of normalization - 1st layer
        x = x + self.mlp(self.ln_2(x))  # 2nd layer   mlp is multi layer pereceptron/feed forward network -- individually the tokens contain informations
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 #max sequence length -- Number of tokens in the input sequence being processed at once.
    vocab_size: int = 50257 #number of tokens -- in LLM the model tranforms texts to tokens
    n_layer: int = 12 #number of layers
    n_head: int = 12 #number of heads -- used for better quality, increasing heads to optimal amount is beneficial, if we increase it properly it would learn more properly parallely and have more ways of thinking capability
    n_embd: int = 768 #embedding dimension

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(    #just replicating the same transformer that we saw when we took it from hugging face
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head =  nn.Linear(config.n_embd, config.vocab_size, bias=False)
    

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
        'gpt2': dict(n_layer=12, n_head=12, n_embd=768), # 124M params
        'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
        'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
        'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        } [model_type]
        config_args ['vocab_size' ] = 50257 # always 50257 for GPT model checkpoints
        config_args ['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig( ** config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys ()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias') ] # discard this mask
        # init a huggingface/transformers model
        model_hf=GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf=model_hf.state_dict()

        sd_keys_hf=sd_hf.keys()
        sd_keys_hf = [k for k in sd_hf if not k.endswith('.attn.masked_bias')] # ignore these
        sd_keys_hf = [k for k in sd_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla nn.Linear.
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys)
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

model=GPT.from_pretrained('gpt2')
print("Hey! It didn't crash")