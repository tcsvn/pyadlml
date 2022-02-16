import sys

from torch.autograd import Variable

from pyadlml.feature_extraction import TimeDiffExtractor

sys.path.append("../")
from pyadlml.dataset import set_data_home, fetch_amsterdam
set_data_home('/tmp/pyadlml_data_home2')
data = fetch_amsterdam(cache=True, retain_corrections=False)
from timeit import default_timer as timer

from pyadlml.preprocessing import StateVectorEncoder, SequenceSlicer, LabelEncoder, Df2Torch, DropTimeIndex
import numpy as np
#res = '10s'
dsize = 2000

#start = timer()
enc = StateVectorEncoder(encode='raw')#, t_res=res)
raw = enc.fit_transform(data.df_devices[:dsize])

lbl_enc = LabelEncoder(idle=True)
lbls = lbl_enc.fit_transform(data.df_activities, raw)
X, y = DropTimeIndex().fit_transform(raw, lbls)

X, y = Df2Torch().fit_transform(X, y)

sc = SequenceSlicer(rep='many-to-many', window_size=4, stride=5)
X, y = sc.fit_transform(X, y)

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMNetwork(nn.Module):
    def __init__(self, input_size, n_classes, hidden_size=100, hidden_layers=1):
        super().__init__()
        self.num_layers = hidden_layers
        self.hidden_size = hidden_size
        self.lstm_layers = []
        for i in range(hidden_layers):
            self.lstm_layers.append(
                nn.LSTMCell(input_size=input_size, hidden_size=hidden_size))

        self.dense = nn.Linear(hidden_size, n_classes)


    def forward(self, x):
        outputs = []

        # initalize the hidden states
        h_0 = torch.zeros(x.shape[0], self.hidden_size)
        c_0 = torch.zeros(x.shape[0], self.hidden_size)
        h_t, c_t = h_0, c_0

        for input_t in x.split(1, dim=1):
            h_t, c_t = self.lstm_layers[0](input_t, (h_t, c_t))
            out = self.relu(h_t)
            out = self.dense(out)
            out = torch.softmax(out, dim=-1)
            outputs += [out]

        outputs = torch.cat(outputs, dim=1)
        return outputs


num_epochs = 1000       # 1000 epochs
learning_rate = 0.001   # 0.001 lr

input_size = 14     # number of features
hidden_size = 20    # number of features in hidden state
num_layers = 1      # number of stacked lstm layers
num_classes = 8     # number of output classes


lstm1 = LSTMNetwork(input_size, num_classes, hidden_size, num_layers)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
  outputs = lstm1.forward(X) #forward pass
  optimizer.zero_grad() #calculate the gradient, manually setting to 0

  # obtain the loss function
  loss = criterion(outputs, y)

  loss.backward() #calculates the loss of the loss function

  optimizer.step() #improve from loss, i.e backprop
  if epoch % 100 == 0:
    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

#end = timer()
#td = end - start
#print('size {} and res {} => elapsed time: {:.3f} seconds'.format(dsize, res, td)) # Time in seconds, e.g. 5.38091952400282