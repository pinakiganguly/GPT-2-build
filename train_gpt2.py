#github access token - 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
from dataclasses import dataclass
from torch.cuda import is_available
from transformers import GPT2LMHeadModel
import math
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken # from openAI
import time
import sys
from torch.nn.parallel import DistributedDataParallel as DDP

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

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
        self.c_proj.NANOGPT_SCALE_INIT = 1  #crude way to reduce the variance of weights of this submodule
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
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) #here the masking is done so that the model cannot the see the future tokens, if so then it will be cheating and the model would not think of itself.
        # att = F.softmax(att, dim=-1) #this generally makes the probability of the raw scores, hence the probability distribution influences how each token influences the current token.
        # # att = self.attn_dropout(att)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs) --> then applying attention to values

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  #here we are applying the Flash attention just to optimize the matrices. Also doing it explicitly for pytorch to see it as flash attention.
        #is_causal is the parameter which is used in gpt models just for masking and asking the models just to see the previous tokens and not thye future ones.

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        # y = self.resid_dropout(self.c_proj(y))
        y = self.c_proj(y)
        return y

class Block(nn.Module):
    """ an unassuming Transformer block with all the submodules """

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

        #weight tying scheme applied - just to reduce the parameters , by using the same space for both the embeddings, which fine tunes the model for getting the similar learning space ---- meaning the input embedding matrix and output projection layer share the same weights.
        self.transformer.wte.weight=self.lm_head.weight
      
        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):   #initialization of the weights accross all submodules that are present inside the nn module
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
              std*= (2* self.config.n_layer) ** -0.5  #here its 2 times because in each layer of the transformer there are 2 paths -> 1 is attention and another is MLP
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  # bias is generally used for giving the model flexibility, its not default value for pytorch
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    # before generating we need to forward it and will feed the forward function the token indices(idx)
    def forward(self, idx, targets=None):
        #idx is of shape(B,T) -> B is batch dimension and T is the time dimension
        B, T = idx.size()  #tokens are in sequences and these sequences are again stacked in batches for effcient computation.
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"  #here block size is the sequece length, B and T are in a 2d space and each lenght of row is <= to the max sequennce length.
        # forward the token and position embeddings
        pos=torch.arange(0, T, dtype=torch.long, device=idx.device) #shape (T)
        pos_emb=self.transformer.wpe(pos)
        tok_emb=self.transformer.wte(idx)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x=block(x)
        # forward the final layernorm and the classifier
        x=self.transformer.ln_f(x)
        logits=self.lm_head(x) # (B, T, vocab_size) #Here the model makes probability to find out which tokens comes next and predicts them.
        loss = None
        if targets is not None:
          loss = F.cross_entropy(logits.view(-1, logits.size(-1)),targets.view(-1)) #cross_entropy does not like multi dimensional inputs so its just breaking down the 3d logits to 2 dimensional and also transforming the targets to singe dimensional tesnor.
          #here we get the loss ~10.9 and its expected to give around 10.8 = -ln(1/cab_size), so we can say roughly we are good with the initialization
        return logits, loss
    

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

    def configure_optimizers(self, weight_decay, learning_rate, device): 
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2] # -- 1D tensors
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]                    #from line 220 to 230 we are just reducing the weights by using the weight decay parameter, so that the training can be done with more stability. Then finally printing out those.
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # if master_process:
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters  #new fused= parameter added my pytorch for AdamW optimizer
        use_fused = fused_available and "cuda" in device  #fused is much faster operation that can be implemented using cuda. Instead of iterating through the for loop for all the tensors, it generates lot of kernels for computation, after the computation is done it combines all into one kernel and finally updates it with all the parameters (kernel fusion update)
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

#-----------------------------------------------------------------------------------------------------

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        with open('/content/GPT-2-build/input.txt','r') as f:
          text = f.read()

        enc=tiktoken.get_encoding('gpt2')
        tokens=enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"l epoch = {len(self.tokens) // (B*T)} batches")

        self.current_position= self.B * self.T * self.process_rank

        # get the shard filenames
    def next_batch(self):  #creating batches in B*T
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1] # B*T+1 just to map with the inputs with target when we crated the tensors for the input and the target.
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets   --- refer inputs and outputs cell in play.ipynb
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes #we are using num_processes because now we will use multiple GPUs which are again ranked for parallelization and so the data loader loads the data accordingly
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):   # --- refer inputs and outputs cell in play.ipynb
            self.current_position=self.B * self.T * self.process_rank #if we are just running out of data we can again reinitialize back to 0
        return x, y

# torchrun --standalone --nproc_per_node=8 train_gpt2.py
from torch.distributed import init_process_group, destroy_process_group
# from torch.nn.parallel import DistributedDataParallel as DDP
# import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank                               --------------------|
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"                                                          #|
    init_process_group(backend='nccl')                                                                                                #|
    ddp_rank = int(os.environ['RANK']) #rank is used to rank the GPUs and to ensure that none of the GPUs get same data for processing |
    ddp_local_rank = int(os.environ['LOCAL_RANK']) #ranking each node in GPU where the data will be processed                          |
    ddp_world_size = int(os.environ['WORLD_SIZE'])  #number GPUs that will be used for parallel processing                             |-----> This is only effective when you have multiple GPUs in your machine (here we took 8 GPUs to process in parallel)
    device = f'cuda:{ddp_local_rank}' #to ensure there is no collisions between GPUs for each process.                                 |
    torch.cuda.set_device(device)                                                                                                     #|
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.                                 -----------------|
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
  torch.manual_seed(1337)



# device = 'cpu' #override

#-------was just exploring with small dataset---------------------------------------------------------------------------------------|
# enc=tiktoken.get_encoding('gpt2')                                                                                                 |
# with open('/content/GPT-2-build/input.txt','r') as f:                                                                             |
#   text = f.read()                                                                                                                 |
# text = text[:1000]  #taking only the 1st 1000 tokens to tsrat with model training                                                 |
# tokens=enc.encode(text)                                                                                                           |------> This whole thing is now in DataLoaderLite class
# B, T = 4, 32        #creating this tensor dimension for the batch layer -- smaller dimension created just for easy debugging      |
# buf = torch.tensor(tokens[:B*T+1]) #here the buf resides in CPU not to GPU                                                        |
# buf = buf.to(device) #we cannot just do .to() because buff itself takes a new memory in the CPU so needs to be reinitialized      |
# x=buf[:-1].view(B,T)                                                                                                              |
# y= buf[1:].view(B,T)                                                                                                              |
#-----------------------------------------------------------------------------------------------------------------------------------|

total_batch_size = 524288 # 2**19 ~0.5M, in number of tokens
B = 2
T = 1024
assert total_batch_size % (B*T*ddp_world_size) == 0 , "make sure total batch size is divisible by B*T*ddp_world_size"
grad_accum_steps = total_batch_size // (B*T*ddp_world_size)
if master_process:
  print(f"Total desired batch size:{total_batch_size}")
  print(f"=> calculated  gradient  accumulation steps: {grad_accum_steps}")

# print("I am GPU:", ddp_rank)   |
# print("bye")                   |-----> Just simulating to see how many GPUs are getting allocated and how the tasks are getting allocated.
# import sys; sys.exit(0)        |

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)

torch.set_float32_matmul_precision('high') # --- we are going to do all the matrix multiplications using tensor float 32 in pytorch not float 32 that we were using before and will run tensor cores of GPU

#creating model
model=GPT(GPTConfig(vocab_size=50304))
model.to(device)
model=torch.compile(model) #here python doesn't read line one by one, here the pytorch takes whole module at a time and compute it in 1 go, which makes the code to run more efficiently
# logits, loss = model(x,y) # passing the labels as well to calculate the loss
if ddp:
  model=DDP(model, device_ids=[ddp_local_rank])

#Now we will perform the gradient and optimize the model and decrease the loss

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50 
def get_lr(it):   #defining the learning rate
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95) , eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

for step in range(max_steps):
  t0 = time.time()
  loss_accum = 0.0                          
  optimizer.zero_grad() #always initialize the gradients to zer before optlimizing #               ---------------------|
  for micro_step in range(grad_accum_steps):  #  after accumulating the gradient for faster computation                 |
    x, y = train_loader.next_batch()   #                                                                                |
    x, y = x.to(device), y.to(device)  #transfering tokens from CPU to GPU                                              |
    with torch.autocast(device_type=device, dtype=torch.bfloat16):                                                   #  |
      logits, loss = model(x, y)  #                                                                                     |
    loss = loss/grad_accum_steps # this is just accumulation of loss at every step                                      |
    loss_accum+= loss.detach()  #   here we are finally calculating the total loss and finally detaching it from model  |
    loss.backward() #applies the gradients whenever there is a loss                                                  #  |---> These are all the tasks that are being sent by CPU and are queued in GPU
  norm = torch.nn.utils.clip_grad_norm(model.parameters(),1.0) #used for stable training and prevent exploding gradients|
  lr = get_lr(step)                                                                                                  #  |
  for param_group in optimizer.param_groups: #Here we are just setting the learning rate                                |
    param_group['lr'] = lr                                                                                            # |
  optimizer.step() # update the parameters and decrease the loss                                  ----------------------|
  torch.cuda.synchronize() #CPU sends instructions and schedules task in GPU, sometimes CPU doesn't track whether the task is completed by GPU, this line of code just make the task synchronized between CPU and GPU
  t1 = time.time()
  dt = (t1-t0)*1000 #time difference in milisecond
  tokens_per_sec = train_loader.B * train_loader.T * grad_accum_steps
  print(f"step {step}| loss : {loss_accum.item()}| lr:{lr:.4e} | norm:{norm: .4f} | dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}") #Here as we know loss is 1 d tensor & stored in GPU and convert it into float and store again to the CPU



# print(loss)
sys.exit(0)

num_return_sequences = 5
max_length = 30

# model=GPT.from_pretrained('gpt2')  #gpt2 model takes all the functionalities that are required by itself like the pretrained function, forward and other functions thata are reuqired for text generation.
# print("Hey! It didn't crash")
model.eval()
model.to(device) #just to shift the running environment from CPU to GPU

enc = tiktoken.get_encoding('gpt2') # tokenizer for gpt 2
tokens = enc.encode("Hello, I am a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) #after tokenizing the whole text also we get 8 tokens, thats the thing its doing
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # Replicating that 8 tokens 5 times, just to match the num_sequence and get 5 different o/p
x = tokens.to(device)

# generate! right now x is (B, T) where B = 5, T = 8
# set the seed to 42
torch.manual_seed(42) # just to ensures your model’s randomness is repeatable, so results are consistent across runs.
torch.cuda.manual_seed(42)
while x.size(1) < max_length: #here we will add new indices to each row of the sequence i.e, meaning we are adding new texts in the existing text that we passed after transforming it into 5 x 8 dim matrix that we created.
    # forward the model to get the logits
    with torch.no_grad():   #this no_grad module of pytorch is generally declared not to keep any cache
        logits = model(x) # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[ :, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim =- 1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim =- 1) #we are taking top 50 probabilities and just ingoring and normalizing other tokens below 50 to 0 - just to keep the model on track
        # select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens) # decoding the tokens back to its string format
    print(">", decoded)
