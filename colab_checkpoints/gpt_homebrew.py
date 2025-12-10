from dataclasses import dataclass
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# configuration class
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257  # GPT-2 vocab size -- 50,000 BPE merges + 256 bytes + <|endoftext|>
    n_layer: int = 12
    n_head: int = 12
    n_emb: int = 768


class LoRALayer(nn.Module):
    def __init__(self, original_linear, rank=8, alpha=None):
        super().__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha if alpha is not None else 2 * rank
        self.A = nn.Parameter(torch.empty((original_linear.in_features, rank)))
        self.B = nn.Parameter(torch.empty((rank, original_linear.out_features)))
        
        # initialize A and B -- using default init for linear layers
        self.A = nn.init.uniform_(self.A, -1/math.sqrt(original_linear.in_features), 1/math.sqrt(original_linear.in_features))
        self.B = nn.init.zeros_(self.B)  # init B w/ zeros enables first passes to have identical behavior as oriiginal linear layer
                                         # this is crucial for retaining pre-trained knowledge at the start of fine-tuning
        self.scaling = self.alpha / self.rank

        # freeze original linear layer
        for param in self.original_linear.parameters():
            param.requires_grad = False

    def forward(self, x):
       return self.original_linear(x) + (x @ self.A @ self.B) * self.scaling
       

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        # ensure that we can evenly split the embedded input across the number of heads
        assert config.n_emb % config.n_head == 0, "Embedding Space not evenly divisible amongst attention heads"

        self.c_attn = nn.Linear(config.n_emb, 3 * config.n_emb)  # for query, key, value -- will split up into distinct K,Q,V Linear Operators later
        self.c_proj = nn.Linear(config.n_emb, config.n_emb)      # "projection" layer
        self.c_proj.NANOGPT_SCALE_INIT =  1  # assign flag to scale residual connection by 1/sqrt(n)

        self.n_head = config.n_head
        self.n_emb = config.n_emb

        # causal mask to ensure that attention is only applied to the left in the input sequence
        # a buffer is a persistent tensor attached to the module that is not a parameter
        # registering here will save it and load it with the model state dict -- maybe this is done to avoid storing -inf weights?
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_emb)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)  # (B, T, 3 * C)
        q, k, v = qkv.split(C, dim=2)  # split into query, key, value -- each (B, T, C)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # each is (B, nh, T, hs) where nh:=n_head and hs:=head_size

        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))     # QK attention with standardization -- (B, nh, T, T)
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))  # apply masking using the buffer
        # att = F.softmax(att, dim=-1)
        # y = att @ v  # (B, nh, T, T) @ (B, nh, T, hs) --> (B, nh, T, hs)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # FlashAttention
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side --> (B, T, C)

        y = self.c_proj(y)  # final projection layer
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_emb, 4 * config.n_emb)  # 4* expansion comes from GPT2 paper
        self.gelu = nn.GELU(approximate='tanh')  #  Gaussian Error Linear Unit -- x * P(X <= x) where X ~ N(0,1) or 0.5x(1+tanh[sqrt(2/pi)(x + 0.044715x^3)]) w/ tanh approx
        self.c_proj = nn.Linear(4 * config.n_emb, config.n_emb)
        self.c_proj.NANOGPT_SCALE_INIT =  1  # assign flag to scale residual connection by 1/sqrt(n)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    

# structure of decoder block from GPT2 paper
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_emb)   # (B, T, C)
        self.attn = CausalSelfAttention(config)  # (B, T, C)
        self.ln_2 = nn.LayerNorm(config.n_emb)   # (B, T, C)
        self.mlp = MLP(config)                   # (B, T, C)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # note that this includes a residual connection (via x + ...)
        x = x + self.mlp(self.ln_2(x))   # this residual connection preserves gradients from Block output back to Block input
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_emb),  # token embeddings
            wpe = nn.Embedding(config.block_size, config.n_emb),  # position embeddings
            h = nn.ModuleList([Block(config) for layer in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_emb)
        ))
        self.lm_head = nn.Linear(config.n_emb, config.vocab_size, bias=False)

        # weight tying -- the token embedding weights are shared with the output projection weights
        # from "Attention is All You Need", and used in GPT-2
        # intuition is that the layers should behave the same, and we can save memory by sharing weights
        self.transformer.wte.weight = self.lm_head.weight  # sends param count ~164M --> ~124M for GPT-2

        self.apply(self._init_weights)  # initialize weights

    # precise weight initializations following GPT2
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5  # scale residual connections
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence length {T} > model block size {self.config.block_size}."

        # token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)  # (T, n_emb)
        tok_emb = self.transformer.wte(idx)  # (B, T, n_emb)
        x = tok_emb + pos_emb   # there is implicit broadcasting here -- (B, T, n_emb) + (T, n_emb) --> (B, T, n_emb)

        # forward pass through transformer blocks
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)  # final layer norm

        logits = self.lm_head(x)      # logits for each token in vocab -- (B, T, vocab_size)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


    @classmethod
    def from_pretrained(cls, model_type, lora_rank: int = 0, lora_alpha: int = None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}, "Unsupported model type"
        from transformers import GPT2LMHeadModel

        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_emb=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_emb=1024),
            'gpt2-large':  dict(n_layer=36, n_head=20, n_emb=1280),
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_emb=1600),
        }[model_type]
        config_args.update(dict(block_size=1024, vocab_size=50257))  # common args for all GPT-2 models
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]  # attn.bias not in the pretrained model

        # init huggingface model and get its state dict
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.bias') and not k.endswith('.attn.masked_bias')]

        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
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

        # if using lora, freeze parameters and replace linear layers with LoRA layers
        if lora_rank > 0:
            for name, module in model.named_modules():
                
                for p in module.parameters():
                    p.requires_grad = False  # freeze all parameters initially

                if isinstance(module, nn.Linear) and ('c_attn' in name or 'c_proj' in name):
                    parent_name = '.'.join(name.split('.')[:-1])
                    attr_name = name.split('.')[-1]
                    
                    parent = model.get_submodule(parent_name) if parent_name else model

                    # Replace with LoRA version
                    lora_linear = LoRALayer(module, rank=lora_rank, alpha=lora_alpha)
                    setattr(parent, attr_name, lora_linear)

        return model