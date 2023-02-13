from torch import nn
from pyadlml.model.nn import Tanh, Embedding
from torch.nn import BatchNorm1d, Sequential



class MLP(nn.Module):
    def __init__(self, n_features, window_length, n_embed, n_hidden, n_classes):
        super().__init__()
        self.layers = Sequential( 
            Embedding(n_features, n_embed),         # B, T      -> B, T, emb
            nn.Flatten(),                           # B, T      -> B, T*emb

            nn.Linear(n_embed*window_length, n_hidden, bias=False),
            BatchNorm1d(n_hidden), 
            Tanh(),

            nn.Linear(n_hidden, n_hidden, bias=False),
            BatchNorm1d(n_hidden), 
            Tanh(),

            nn.Linear(n_hidden, n_hidden, bias=False),
            BatchNorm1d(n_hidden), 
            Tanh(),

            #nn.Linear(n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
            nn.Linear(n_hidden, n_classes),         # B, H      -> B, C
        )

    #def parameters(self):
    #    return self.layers
    #    #return [p for layer in self.layers for p in layer.parameters()]

    def __call__(self, X):
        for l in self.layers:
            X = l(X)
        return X

