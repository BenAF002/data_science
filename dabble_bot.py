import nltk
assert(nltk.download('wordnet'))
from nltk.corpus import wordnet as wn
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as a
import time
import string
import pandas as pd
from pathlib import Path


try:
    from google.colab import drive
    drive.mount('/content/drive')
    path = Path('/content/drive/My Drive/Colab Notebooks')
except:
    path = Path.cwd()
    pass

# global vars
device = 'cuda' if torch.cuda.is_available() else 'cpu'

wsample = []
with open(path / 'word_sample.txt', 'r') as file:
  for line in file.read().splitlines():
    wsample.append(line)

# load vocab excl rare words
trim_vocab = []
with open(path / 'trim_vocab.txt', 'r') as file:
  for line in file.read().splitlines():
    trim_vocab.append(line)

# load definitions excl those that include rare words
more_trim_defs = []
with open(path / 'more_trim_defs.txt', 'r') as file:
  for line in file.read().splitlines():
    more_trim_defs.append(line)

end_char = '.'
start_char = '<s>'
pad_char = '<p>'

stoi = {s:i+1 for i,s in enumerate(trim_vocab)}    # word-to-integer mapping dictionary
stoi[end_char] = len(stoi) + 1                     # adding end character
stoi[start_char] = len(stoi) + 2                   # adding start character
stoi[pad_char] = 0                                 # adding pad character
itos = {i:s for s,i in stoi.items()}               # integer-to-word mapping dictionary

vocab_size = len(stoi) + 1
encoder = lambda s: [stoi[c] for c in s]            # encoder
decoder = lambda l: ' '.join([itos[i] for i in l])  # decoder


class Head(nn.Module):
    def __init__(self, n_emb, head_size, dropout, block_size):
        super().__init__()
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))  # (T, T)
            # torch.tril() is not a parameter, so we have to use register_buffer to assign it to the module
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, hs)
        q = self.query(x)  # (B, T, hs)

        # compute attention scores ("affinities")
        weights = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5   # (B, T, T); scaled by 1/sqrt(hs)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        # aggregate values by weights
        v = self.value(x)
        out = weights @ v
        return out

class MultiHead(nn.Module):
    def __init__(self, n_emb, n_heads, head_size, dropout, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_emb, head_size, dropout, block_size) for h in range(n_heads)])  # create list of heads
        self.proj = nn.Linear(head_size * n_heads, n_emb)  # linear transformation of the output from the head stack
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # feed forward through heads and concatenate output
        out = self.dropout(self.proj(out))                   # pass output through linear layer and apply dropout
        return out

class FeedForward(nn.Module):
    def __init__(self, n_emb, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),     # mult 4 bc the paper does a 4x channel expansion in the feedforward
            nn.ReLU(),
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_emb, n_heads, dropout, block_size):
        super().__init__()
        head_size = n_emb // n_heads
        self.sa = MultiHead(n_emb, n_heads, head_size, dropout, block_size)  # self-attention stack
        self.ffwd = FeedForward(n_emb, dropout)   # feed-forward layer
        self.ln1 = nn.LayerNorm(n_emb)   # layer normalization for self-attention stack
        self.ln2 = nn.LayerNorm(n_emb)   # layer normalization for feed forward

    def forward(self, x):
        # x = self.ln1(x + self.sa(x))     # residual self-attention stack connection
        # x = self.ln2(x + self.ffwd(x))   # residual feed-forward connection
        x = x + self.sa(self.ln1(x))     # residual self-attention stack connection
        x = x + self.ffwd(self.ln2(x))   # residual feed-forward connection
        return x


class DabbleBot(nn.Module):
    def __init__(self,
                 words: list,
                 batch_size: int = 128,
                 n_emb: int = 256,
                 n_heads: int = 8,
                 n_blocks: int = 8,
                 dropout: float = 0.2
    ):
        """
        DabbleBot nlp model

        Args:
            words (list): list of words to train on
            batch_size (int, optional): batch size. Defaults to 128.
            n_emb (int, optional): embedding size. Defaults to 256.
            n_heads (int, optional): number of heads. Defaults to 8.
            n_blocks (int, optional): number of blocks. Defaults to 8.
            dropout (float, optional): dropout rate. Defaults to 0.2.
        """
        # input data
        self.words = words
        self.Xt, self.Yt, self.Xv, self.Yv = self.preprocess()

        # loss tracking
        self.tloss, self.vloss = [], []

        # hyperparameters
        self.batch_size = batch_size
        self.n_emb = n_emb
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.block_size = self.max_length + 1

        # functional modules
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb)      # token embedding
        self.position_embedding_table = nn.Embedding(self.block_size, n_emb)   # positional embedding
        self.blocks = nn.Sequential(*[Block(n_emb, n_heads, dropout, self.block_size) for b in range(n_blocks)])
        self.ln_f = nn.LayerNorm(n_emb)                # final layer norm
        self.lm_head = nn.Linear(n_emb, vocab_size)    # output linear layer


    def preprocess(self):
      """
      Preprocess the words to be used in the model
      """
      words = self.words
      definitions = [s.definition() for w in words for s in wn.synsets(w)]
      trim_definitions = [''.join(d).translate(str.maketrans('', '', string.punctuation)) + ' . ' for d in definitions]
      trim_defstring = ''.join(trim_definitions)
      vocab = list(set(sorted(trim_defstring.split())))

      # remove rare words
      counts = [trim_defstring.count(word) for word in vocab]
      rare_words = {vocab[i] for i,c in enumerate(counts) if c < 2}
      more_trim_defs= [d for d in trim_definitions if len(set(d.split()) & rare_words) == 0]

      data = [encoder(d.split()) for d in more_trim_defs]
      self.max_length = max([len(d) for d in data])
      xdat = [encoder([start_char]) + d[:-1] for d in data]
      ydat = [d for d in data]

      # right pad all definitions to max length
      for d in xdat: d += [0] * (self.max_length - len(d) + 1)
      for d in ydat: d += [0] * (self.max_length - len(d) + 1)
      xdat = torch.tensor(xdat)
      ydat = torch.tensor(ydat)

      # produce training and valdation data
      n = int(0.8*len(data))
      Xt, Yt = xdat[:n], ydat[:n]  # 80% training data
      Xv, Yv = xdat[n:], ydat[n:]  # 20% validation data

      return Xt, Yt, Xv, Yv


    def _minibatch(self, xdat, ydat):
      idx = torch.randint(len(xdat) - self.block_size, (self.batch_size,))  # 1D tensor of random ints
      x, y = xdat[idx], ydat[idx]  # index into x and y tensors using random ints
      return x.to(device), y.to(device)


    def forward(self, input, targets = None):
        B, T = input.shape

        # idx and targets are both (B, T) tensor of integers                    dimension tracking:
        tok_emb = self.token_embedding_table(input)                               # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))   # (T,C)
        x = tok_emb + pos_emb                                                     # (B,T,C)
        x = self.blocks(x)                                                        # (B,T,C)
        x = self.ln_f(x)                                                          # (B,T,C)
        logits = self.lm_head(x)                                                  # (B,T,vocab_size)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets, ignore_index=0)    # ignore index of pad_char

        return logits, loss


    @torch.no_grad()
    def estimate_loss(self, check_iters):
        """
        Estimate training and validation set loss
        """
        out = {}
        data = {'train': (self.Xt, self.Yt), 'val': (self.Xv, self.Yv)}
        self.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(check_iters)
            for k in range(check_iters):
                X, Y = self._minibatch(*data[split])
                logits, loss = self.forward(input=X, targets=Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.train()
        return out


    def generate(self, samples, idx = [[stoi[start_char]]]):    # idx is (B, T) array of indices in the current context
        """
        Generate text from the model

        Args:
            samples (int): number of samples to generate
            idx (torch.Tensor, optional): (B, T) array of indices in the current context
        """
        self.eval()
        sample = []

        for s in range(samples):
            ctx = idx

            while True:
                ctx_cond = ctx[:, -self.block_size:]
                logits, loss = self(ctx_cond)

                # focus only on the last time step
                logits = logits[:, -1, :] # becomes (B, C)

                probs = F.softmax(logits, dim=-1) # (B, C)
                ctx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

                # append sampled index to the running sequence
                ctx = torch.cat((ctx, ctx_next), dim=1) # (B, T+1)

                if ctx_next.item() == stoi[end_char] or ctx.shape[1] > 50:
                    break
            sample.append(decoder(ctx.tolist()[0]))

        self.train()

        return sample


    def train_loop(self, lr: float = 1e-3, epochs: int = 10, epoch_iters: int = 1000, check_iters: int = 100):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)  # paper uses Adam optimizer
        max_iters = epochs * epoch_iters
        run_time = 0

        for iter in range(max_iters):
            start = time.time()

            # every once in a while evaluate the loss on train and val sets
            if (iter % check_iters == 0) or (iter == max_iters - 1):
                losses = self.estimate_loss(check_iters)
                self.tloss.append(losses['train'])
                self.vloss.append(losses['val'])
                if (iter % epoch_iters == 0) or (iter == max_iters - 1):
                    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} \
                        | ETA: {run_time / 60 * ((max_iters - iter) + check_iters * ((max_iters - iter) / check_iters)):.2f} min")

            # sample a batch of data
            xb, yb = self._minibatch(self.Xt, self.Yt)

            # evaluate loss
            logits, loss = self.forward(input=xb, targets=yb)
            optimizer.zero_grad(set_to_none=True)  # set zeroed gradients to none to halt descent through those gradients
            loss.backward()
            optimizer.step()

            run_time = time.time() - start