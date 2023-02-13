from torch import nn
from pyadlml.model.nn import Tanh, Embedding, FlattenConsecutive
from torch.nn import Linear, BatchNorm1d
import torch 

def print_in_out(layer, name=None):
    def wrapper(x):
        i = list(x.shape)
        x = layer(x)
        j = list(x.shape)
        name = layer.__class__.__name__ if name is None else name
        print(f'{name} {i} -> {j}')
        return x
    return wrapper

class WaveNet(nn.Module):

    def __init__(self, n_features, n_emb, n_hidden, n_classes, dilation_factor):
        """
        """
        super().__init__()

        d = dilation_factor
        self.l_emb = Embedding(n_features, n_emb)             # B, T            -> B, T, E
        self.l_fltn1 = FlattenConsecutive(d)                   # B, T, E         -> B, T/d, E*d
        self.l_lin1 = Linear(n_emb*d, n_hidden, bias=False)    # B, T/d, E*d     -> B, T/d, H
        self.l_bn1 = BatchNorm1d(n_hidden)                     # B, T/d, H       -> B, T/d, H
        self.l_tanh1 = Tanh()                                  # B, T/d, H       -> B, T/d, H

        self.l_fltn2 = FlattenConsecutive(d)                      # B, T/d, H       -> B, T/d^2, H*d
        self.l_lin2 = Linear(n_hidden*d, n_hidden, bias=False) # B, T/d^2, H*d   -> B, T/d^2, H*d
        self.l_bn2 = BatchNorm1d(n_hidden)                     # B, T/d^2, H*d   -> B, T/d^2, H*d
        self.l_tnh2 = Tanh()                                   # B, T/d^2, H*d   -> B, T/d^2, H*d

        self.l_fltn3 = FlattenConsecutive(d)                      # B, T/d^2, H     -> B, T/d^3, H*d^2
        self.l_lin3 = Linear(n_hidden*d, n_hidden, bias=False) # B, T/d^3, H*d^2 -> B, T/d^3, H*d^2
        self.l_bn3 = BatchNorm1d(n_hidden)                     # B, T/d^3, H*d^2 -> B, T/d^3, H*d^2
        self.l_tnh3 = Tanh()                                   # B, T/d^3, H*d^2 -> B, T/d^3, H*d^2

        self.l_fltn4 = FlattenConsecutive(d)                      # B, T/d^2, H     -> B, T/d^3, H*d^2
        self.l_lin4 = Linear(n_hidden*d, n_hidden, bias=False) # B, T/d^3, H*d^2 -> B, T/d^3, H*d^2
        self.l_bn4 = BatchNorm1d(n_hidden)                     # B, T/d^3, H*d^2 -> B, T/d^3, H*d^2
        self.l_tnh4 = Tanh()                                   # B, T/d^3, H*d^2 -> B, T/d^3, H*d^2

        self.lin_5 = nn.Linear(n_hidden, n_classes)


    @property
    def layers(self):
        return self.parameters()


    def forward(self, x):
        """
        """



        x = self.l_emb(x)
        x = self.l_fltn1(x) 
        x = self.l_lin1(x)  # d -> B, T/d, H
        x = x.swapdims(1,2)
        # Expects (N, C, L) where N=batchsize, C=features, L = seq_length
        x = self.l_bn1(x)
        x = x.swapdims(1,2)
        x = self.l_tanh1(x)

        x = self.l_fltn2(x)
        x = self.l_lin2(x)
        x = x.swapdims(1,2)
        x = self.l_bn2(x)
        x = x.swapdims(1,2)
        x = self.l_tnh2(x)

        x = self.l_fltn3(x)
        x = self.l_lin3(x)
        x = x.swapdims(1,2)
        x = self.l_bn3(x)
        x = x.swapdims(1,2)
        x = self.l_tnh3(x)


        x = self.l_fltn4(x)
        x = self.l_lin4(x)
        #x = x.swapdims(1,2)
        x = self.l_bn4(x)
        #x = x.swapdims(1,2)
        x = self.l_tnh4(x)



        x = self.lin_5(x)
        assert x.dim() == 2, 'Bigrams do not resolve. Choose different numbers'

        return x


