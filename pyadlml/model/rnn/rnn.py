#from skorch.utils import to_tensor
#from skorch.utils import TeeGenerator
#from skorch import NeuralNetClassifier

import torch.nn as nn
import torch.nn.functional as F
import torch

import torch.nn as nn
import torch.nn.functional as F
class RNN(nn.Module):
    def __init__(self, input_size, n_classes, rec_layer_type='lstm',
                 hidden_size=300, hidden_layers=1, seq='many-to-one'):
        super().__init__()
        # set parameters for
        self.input_size = input_size
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.rec_layer_type = rec_layer_type.lower()
        self.seq_type = seq

        self.reset_weights()

    def reset_weights(self):
        # initialize the weights
        rec_layer = {'lstm': nn.LSTM, 'gru': nn.GRU}[self.rec_layer_type]
        #input and ouput are providedd as (batch, seq, feature)
        self.rnn_ = rec_layer(self.input_size,
                            hidden_size=self.hidden_size,
                            batch_first=True)
        self.out_ = nn.Linear(self.hidden_size, self.n_classes)

    def forward(self, x):
        seq_length = x.shape[1]

        # forward pass through recurrent net
        if self.rec_layer_type == 'gru':
            out, _ = self.rnn_(x)
        elif self.rec_layer_type == 'lstm':
            # output (batch_size , seq length, hidden_size)
            # hn (n_layers, batch_size, hidden_size)
            # cn (n_layers, batch_size, hidden_size)
            out, _ = self.rnn_(x)

        # get either the whole latent state sequence or only the last
        if self.seq_type == 'many-to-one':
            # take output of last stacked RNN layer
            # Note: equiv to latent_rep = h_n[-1]
            latent_rep = out[:, seq_length-1, :]
        else:
            latent_rep = out
            # todo flatten the array
        #print('lr: ', latent_rep.shape)
        out = F.softmax(self.out_(latent_rep), dim=-1)
        #print('X:',out.shape)

        return out

#import skorch
#class RegularizedNet(NeuralNet):
#    def __init__(self, *args, lambda1=0.01, **kwargs):
#        super().__init__(*args, **kwargs)
#        self.lambda1 = lambda1
#
#    def get_loss(self, y_pred, y_true, X=None, training=False):
#        loss = super().get_loss(y_pred, y_true, X=X, training=training)
#        loss += self.lambda1 * sum([w.abs().sum() for w in self.module_.parameters()])
#        return loss


#from skorch.utils import to_tensor
#
#class RNNClassifier(NeuralNetClassifier):
#
#    def get_loss(self, y_pred, y_true, X=None, training=False):
#        """Return the loss for this batch.
#        Parameters
#        ----------
#        y_pred : torch tensor
#          Predicted target values
#        y_true : torch tensor
#          True target values.
#        X : input data, compatible with skorch.dataset.Dataset
#        """
#        if self.module_.seq_type == 'many-to-many':
#            y_true = torch.flatten(to_tensor(y_true, device=self.device), start_dim=0, end_dim=1)
#            y_pred = torch.flatten(y_pred, start_dim=0, end_dim=1)
#
#        elif self.module_.seq_type != 'many-to-one':
#            ValueError("the sequence type of the RNN was false defined")
#
#        if isinstance(self.criterion_, torch.nn.Module):
#            self.criterion_.train(training)
#
#        return self.criterion_(y_pred, y_true)