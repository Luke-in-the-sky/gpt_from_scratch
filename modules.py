from torch.nn import (
    Embedding,
    Sequential,
    Linear,
    Tanh,
    Flatten,
    BatchNorm1d,
    ReLU,
    LayerNorm,
    Dropout,
)
from torch.nn import functional as F
from dataclasses import dataclass
import os
import requests
import numpy as np
import torch
from typing import Literal, Callable, Iterable


# Helpers
def download_data(destination_file: str = "input.txt"):
    # download the tiny shakespeare dataset
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    with open(destination_file, "w") as f:
        f.write(requests.get(data_url).text)


@dataclass
class Hyperparams:
    """just a holder for the hyperparams"""

    # input dimensions
    block_size: int = 16
    batch_size: int = 64
    embedding_dim: int = 2**5 * 3
    # transformer params
    num_transf_blocks: int = 1
    num_heads: int = 6
    dropout_rate: int = 0.2
    # trainin params
    learning_rate: float = 1e-3
    training_steps: int = 5000
    # others
    vocab_size: int = None


@dataclass
class DataSet:
    train: Iterable
    validation: Iterable


@dataclass
class DataLoader:
    block_size: int
    batch_size: int
    data: DataSet

    def get_batch(self, split: Literal["train", "val"]):
        segment = self.data.train if split == "train" else self.data.validation
        ix = torch.randint(len(segment) - self.block_size, (self.batch_size,))
        x = torch.stack([segment[i : i + self.block_size] for i in ix])
        y = torch.stack([segment[i + 1 : i + 1 + self.block_size] for i in ix])
        return x, y


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, loader: DataLoader, num_evals: int = 100):
    """
    uses `loader.get_batch` to grab a batch from each of train and test (`num_evals` times)
    and calls `model` on the batches to get the loss

    returns a dict of list of loss values
    """
    model.eval()
    out = {}
    for split in ["train", "validation"]:
        losses = [
            # `model()` returns the tuple `(logits, loss)`
            model(*loader.get_batch(split))[1]
            for i in range(num_evals)
        ]
        out[split] = torch.tensor(losses).mean()
    model.train()
    return out


# Modules


# @dataclass decorator will add an __eq__() method to your class,
# which in turn requires Python to use the __hash__() method. However,
# since nn.Module does not define a __hash__() method (as instances of
# nn.Module are mutable and thus should not be hashable), we will get an
# error. But if you do `eq=False`, no _eq__() method will be added
@dataclass(eq=False)
class SelfAttention(torch.nn.Module):
    emb_dim: int
    head_dim: int
    dropout_rate: int = 0.0

    def __post_init__(self):
        super().__init__()
        # self attention
        self.K = Linear(self.emb_dim, self.head_dim, bias=False)
        self.Q = Linear(self.emb_dim, self.head_dim, bias=False)
        self.V = Linear(self.emb_dim, self.head_dim, bias=False)

        self.dropout = Dropout(self.dropout_rate)

    def __call__(self, x):
        B, T, C = x.shape
        k = self.K(x)  # shape (B, T, C) @ (C, H) = (B, T, H)
        q = self.Q(x)  # shape (B, T, C) @ (C, H) = (B, T, H)
        v = self.V(x)  # shape (B, T, C) @ (C, H) = (B, T, H)
        affin = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, H) @ (B, H, T) = (B, T, T)

        # avg over preceeding tokens
        ## we also initialize the `trill` matrix, but call it a buffer so pytorch does not optimize it
        self.register_buffer("tril", torch.tril(torch.ones((T, T))))
        ## where `tril` is zero, replace with `float(-inf')`
        affin = affin.masked_fill(self.tril == 0, float("-inf"))  # (B, T, T)
        self.affin = F.softmax(
            affin, dim=-1
        )  # normalize affinities along dim 1; shape = (B, T, T)
        self.affin = self.dropout(
            self.affin
        )  # dropout: some are not allowed to talk to each other

        # compute output
        self.out = self.affin @ v  # (B, T, T) @ (B, T, H) = (B, T, H)
        return self.out


@dataclass(eq=False)
class MultiHeadAttention(torch.nn.Module):
    """
    just a list of SelfAttention modules; outputs are concatenated.

    note that we want the out channel dimension to be the same as emb_dim
    so that we can build residual connecitons to the input
    """

    emb_dim: int
    num_heads: int
    dropout_rate: int = 0

    def __post_init__(self):
        super().__init__()
        assert (
            self.emb_dim % self.num_heads == 0
        ), "emb_dim needs to be divisible by num_heads"
        self.head_dim = self.emb_dim // self.num_heads
        self.heads = torch.nn.ModuleList(
            [
                SelfAttention(
                    self.emb_dim, self.head_dim, dropout_rate=self.dropout_rate
                )
                for _ in range(self.num_heads)
            ]
        )
        self.proj = Sequential(
            Linear(self.emb_dim, self.emb_dim),  # TODO: why do we need this??
            Dropout(self.dropout_rate),
        )

    def forward(self, x):
        """apply heads, concat results"""
        # x.shape = (B, T, C - aka emd_dim)
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        # out.shape = (B, T, H, num_heads) --> (B, T, H*num_heads) = (B, T, emb_dim/num_head * num_heads) = (B, T, emb_dim)
        out = self.proj(out)
        return out


@dataclass(eq=False)
class FeedForward(torch.nn.Module):
    """a Linear followed by a activation function"""

    dim_in: int
    dim_out: int
    dropout_rate: int = 0

    def __post_init__(self):
        super().__init__()
        self.model = Sequential(
            Linear(
                self.dim_in, 4 * self.dim_out
            ),  # the Attention paper suggested to have a 4x increase inside the FFwd
            ReLU(),
            Linear(4 * self.dim_in, self.dim_out),
            Dropout(self.dropout_rate),
        )

    def forward(self, x):
        return self.model(x)  # out.shape = (*x.shape[:1], dim_out)


@dataclass(eq=False)
class TransformerBlock(torch.nn.Module):
    """
    multihead with residual+g_normalize; feedforward with residual+g_normalize
    intuition: communication followed by computation
    """

    emb_dim: int
    num_heads: int
    dropout_rate: int = 0
    # we don't need a `dim_out` param: we want residual connections, so we
    # need our feed-forward to output the same dimensionality as our input

    def __post_init__(self):
        super().__init__()
        self.multih = MultiHeadAttention(
            emb_dim=self.emb_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )
        self.ffwd = FeedForward(
            self.emb_dim, self.emb_dim, dropout_rate=self.dropout_rate
        )
        self.lnorm1 = LayerNorm(self.emb_dim)
        self.lnorm2 = LayerNorm(self.emb_dim)

    def forward(self, x):
        # multihead, residual;
        x = x + self.multih(self.lnorm1(x))  # out.shape = (B, T, emb_dim)
        # FFwd, residual;
        x = x + self.ffwd(self.lnorm2(x))  # out.shape = (B, T, emb_dim)
        return x


# Network


@dataclass(eq=False)
class MyGPT(torch.nn.Module):
    hypers: Hyperparams

    def __post_init__(self):  # `post_init` is the pattern for `@dataclass`
        super().__init__()

        # checks
        assert self.hypers.vocab_size > 0, "No vocab_size set"

        # embeddings
        self.token_emb = Embedding(
            self.hypers.vocab_size, self.hypers.embedding_dim
        )  # out.shape = (B, T, C)
        self.pos_emb = Embedding(
            self.hypers.block_size, self.hypers.embedding_dim
        )  # out.shape = (B, T, C)

        # transformer
        self.blocks = Sequential(
            *[
                TransformerBlock(
                    emb_dim=self.hypers.embedding_dim,
                    num_heads=self.hypers.num_heads,
                    dropout_rate=self.hypers.dropout_rate,
                )
                for _ in range(self.hypers.num_transf_blocks)
            ]
        )  # out.shape = (B, T, C)
        self.layer_norm = LayerNorm(
            self.hypers.embedding_dim
        )  # because the out of TransformerBlock is not normalized

        # final for softmax
        self.lm_head = Linear(self.hypers.embedding_dim, self.hypers.vocab_size)

    def forward(self, ix, targets=None):
        B, T = ix.shape

        # forward pass
        tok_emb = self.token_emb(ix)  # out.shape = (B, T, C)
        pos_emb = self.pos_emb(torch.arange(T))  # out.shape =    (T, C)
        x = tok_emb + pos_emb  # out.shape = (B, T, C)  note: broadcast

        att = self.blocks(x)  # out.shape = (B, T, C)
        att_normalized = self.layer_norm(att)
        logits = self.lm_head(att_normalized)  # out.shape = (B, T, vocab_size)

        # compute loss
        if targets is None:  # ie generation
            loss = None
        else:
            # we just want to compute the loss, but pytorch's cross entropy wants the C
            # dimention to be the second dimension instead of the third
            # so we need a little rearrangement
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # out.shape = (B * T, vocab_size)
            targets = targets.view(B * T)  # out.shape = (B)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # note: idx is the current context of token IDs
        # ids.shape = (B, T)
        output = []

        for i in range(max_new_tokens):
            # ensure context is not longer than block_size
            idx = idx[:, -self.hypers.block_size :]

            # get logits, pick only the last one in T dimension (we use max context available for generaiton)
            logits, loss = self(idx)  # logits.shape = (B, T, vocab_size)
            logits = logits[:, -1, :]  # logits.shape = (B, vocab_size)

            # turn into probs and sample
            probs = F.softmax(logits, dim=1)  # probs.shape = (B, vocab_size)
            pred = torch.multinomial(
                probs, num_samples=1, replacement=True
            )  # pred.shape = (B, 1)

            # update the context
            idx = torch.cat((idx, pred), dim=1)[:, 1:]
            output.append(pred.item())
        return output
