import torch
from torch import nn
from torch.nn.parameter import Parameter
import numpy as np


class FlattenConsecutive(nn.Module):
    def __init__(self, d):
        super().__init__()
        """
        Parameters
        ----------
        d : int 
            dilation factor
        """
        self.d = d

    def forward(self, x):
        B, T, E = x.shape   # Batch, Sequence length T, embedding dimension E
        try:
            x = x.view(B, T//self.d, E*self.d)
        except RuntimeError:
            x = x.contiguous().view(B, T//self.d, E*self.d)

        # Do this if dilation is uneven, 
        if x.shape[1] == 1:
            x = x.squeeze(1)

        self.out = x
        return self.out
            
    def parameters(self):
        return []
    

class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn((num_embeddings, embedding_dim)))
    
    def forward(self, IX):
        """
        Plug out the emb vectors for each dev given their indices

        (B, T) ->  (B, T, E)
        """
        self.out = self.weight[IX.long()]
        return self.out
  
    def parameters(self):
        return [self.weight]
    

class Tanh(nn.Module):
    def __call__(self, x):
        super().__init__()
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


class Attention(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        return x


class Time2Vec(nn.Module):
    """ TODO Implememnt

        https://arxiv.org/pdf/1907.05321.pdf
        paper time2vec
    """
    def __init__(self, k):
        pass
        # TODO Implement
        
class RelativePositionalEncoding(nn.Module):
    def __init__(self, emb_dim):
        raise NotImplementedError
        # TODO implement

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=4096):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):

        aa = len(x.size())
        if aa > 1:
            length = x.size(1)
        else:
            length = x.size(0)

        return self.pe[:, :length]


    @classmethod
    def plot(cls, T=300, max_pos=None):
        D = 120
        T = 300
        
        D = int(np.ceil(D / 2) * 2)
        
        from torch import tensor
        inv_freq = 1.0 / (10000 ** (torch.arange(0, T, 2).float() / T))
        pos_x = torch.arange(T, dtype=torch.float32)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
        emb_x = cls.pos_encoding(sin_inp_x)

        emb = torch.zeros((T, D), dtype=torch.float32)
        #emb[:, : D] = emb_x


        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.pcolormesh(emb_x, cmap='copper')
        ax.set_xlabel('Embedding length')
        ax.set_xlim((0, D))
        ax.set_ylabel('Position $t$ in Sequence $1:T$')
        ax.set_title("Positional Encoding $M_{ij}$")
        return fig


import math
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)