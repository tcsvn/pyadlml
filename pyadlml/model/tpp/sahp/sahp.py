'''
self-attentive Hawkes process
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math, copy

from .embedding.event_type import TypeEmbedding
from .embedding.position import PositionalEmbedding,BiasedPositionalEmbedding
from .embedding.event_embedding import EventEmbedding
from .attention.multi_head import MultiHeadedAttention
from .utils.sublayer import SublayerConnection
from .utils.feed_forward import PositionwiseFeedForward
from .base import SeqGenerator, predict_from_hidden
from .utils.gelu import GELU

from matplotlib import pyplot as plt


# TODO remove method and class and reuse transformer masking from VanillaTransformer
def subsequent_mask(size):
    "mask out subsequent positions"
    atten_shape = (1,size,size)
    # np.triu: Return a copy of a matrix with the elements below the k-th diagonal zeroed.
    mask = np.triu(np.ones(atten_shape),k=1).astype('uint8')
    aaa = torch.from_numpy(mask) == 0
    return aaa

class MaskBatch():
    "object for holding a batch of data with mask during training"
    def __init__(self,src,pad, device):
        self.src = src
        self.src_mask = self.make_std_mask(self.src, pad, device)

    @staticmethod
    def make_std_mask(tgt,pad,device):
        "create a mask to hide padding and future input"
        # torch.cuda.set_device(device)
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)).to(device)
        return tgt_mask

def one_hot_embedding(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Embedding labels to one-hot form. Produces an easy-to-use mask to select components of the intensity.
    Args:
        labels: class labels, sized [N,].
        num_classes: number of classes.
    Returns:
        (tensor) encoded labels, sized [N, #classes].
    """
    device = labels.device
    y = torch.eye(num_classes).to(device)
    return y[labels]



class SAHP(nn.Module):
    "Generic N layer attentive Hawkes with masking"

    def __init__(self, nLayers, d_model, atten_heads, dropout, process_dim, device, max_seq_len):
        super(SAHP, self).__init__()
        self.nLayers = nLayers
        self.process_dim = process_dim  # # different types of events
        self.input_size = process_dim + 1
        self.query_size = d_model // atten_heads
        self.device = device
        self.gelu = GELU()

        self.d_model = d_model
        self.type_emb = TypeEmbedding(self.input_size, d_model, padding_idx=self.process_dim)
        self.position_emb = BiasedPositionalEmbedding(d_model=d_model,max_len = max_seq_len)

        self.attention = MultiHeadedAttention(h=atten_heads, d_model=self.d_model)
        self.feed_forward = PositionwiseFeedForward(d_model=self.d_model, d_ff=self.d_model * 4, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=self.d_model, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=self.d_model, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

        self.start_layer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=True),
            self.gelu
        )

        self.converge_layer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=True),
            self.gelu
        )

        self.decay_layer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=True)
            ,nn.Softplus(beta=10.0)
        )

        self.intensity_layer = nn.Sequential(
            nn.Linear(self.d_model, self.process_dim, bias = True)
            ,nn.Softplus(beta=1.)
        )

    def state_decay(self, converge_point, start_point, omega, duration_t):
        # Hawkes process
        # Tau is timedifferences from t_i
        # tanh[(t_i - t)e^(-\omega * \tau)]
        """
        Returns
        -------
        cell_t
            (B, T, E)
        """ 
        # * element-wise product
        cell_t = torch.tanh(converge_point + (start_point - converge_point) * torch.exp(- omega * duration_t))
        return cell_t

    def forward(self, X):
        """
        seq_dt: B, S, L
        seq_types: 
        src_mask: 
        """
        B, S, E = X.shape
        # TODO refactor, emulate input
        batch_seq_times, batch_seq_types = X[:, :, 0], X[:, :, 1]
        batch_seq_types_ = batch_seq_types[:, 1:]
        masked_seq_types = MaskBatch(batch_seq_types_, pad=self.process_dim, device=self.device)# exclude the first added even
        src_mask = masked_seq_types.src_mask
        batch_seq_types_ = batch_seq_types[:, :-1]
        seq_dt, seq_types = batch_seq_times[:, :-1], batch_seq_types[:, :-1].long() 

        # (B, S) -> (B, S, E) with E = embedding dimension 
        type_embedding = self.type_emb(seq_types) * math.sqrt(self.d_model)  #
        position_embedding = self.position_emb(seq_types,seq_dt)

        x = type_embedding + position_embedding
        for i in range(self.nLayers):
            x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=src_mask))
            x = self.dropout(self.output_sublayer(x, self.feed_forward))

        # Latent representation or history (B, S, E)
        embed_info = x

        # Estimate latent parameters for each sequence (B, S, E)
        self.start_point = self.start_layer(embed_info)
        self.converge_point = self.converge_layer(embed_info)
        self.omega = self.decay_layer(embed_info)


        #return logits

    def compute_loss(self, X, n_mc_samples = 20):
        """
        Compute the negative log-likelihood as a loss function.

        Args:
            seq_times: event occurrence timestamps
                (B, S)
            seq_onehot_types: types of events in the sequence, one hot encoded
                (B, S, T) for types i.e. [0, 1] used for masking the relevant events in the sequence
            batch_sizes: batch sizes for each event sequence tensor, by length
            tmax: temporal horizon

        Returns:
            log-likelihood of the event times under the learned parameters

        Shape:
            one-element tensor
        """

        seq_times, batch_seq_types = X[:, :, 0], X[:, :, 1].long()

        # Compute onehot
        batch_onehot = one_hot_embedding(batch_seq_types, self.input_size)
        batch_onehot = batch_onehot[:, :, :self.process_dim]# [1,0], [0,1], [0,0]
        seq_onehot_types = batch_onehot


        # Calculate timedifference tau_i = t_(i+1) -t_i; (B, S) -> (B, S-1) 
        dt_seq = seq_times[:, 1:] - seq_times[:, :-1]
        # (B, S, E)
        cell_t = self.state_decay(self.converge_point, self.start_point, self.omega, dt_seq[:, :, None])


        n_batch = seq_times.size(0)
        n_times = seq_times.size(1) - 1
        device = dt_seq.device
        # Get the intensity process
        # (D, S, T) for each event type compute the intensity at the events
        intens_at_evs = self.intensity_layer(cell_t)
        intens_at_evs = nn.utils.rnn.pad_sequence(
            intens_at_evs, padding_value=1.0,batch_first=True)  # pad with 0 to get rid of the non-events, log1=0
        log_intensities = intens_at_evs.log()  # log intensities
        seq_mask = seq_onehot_types[:, 1:]
        log_intensities_ret = (log_intensities * seq_mask).sum(dim=2)
        log_sum = (log_intensities * seq_mask).sum(dim=(2, 1))  # shape batch

        # Sample for each batch and sequence random inter-event times
        # (B, S, 1, M) where M stands for #markov samples
        taus = torch.rand(n_batch, n_times, 1, n_mc_samples).to(device)# self.process_dim replaced 1

        # 
        taus = dt_seq[:, :, None, None] * taus  # inter-event times samples)

        cell_tau = self.state_decay(
            self.converge_point[:,:,:,None],
            self.start_point[:,:,:,None],
            self.omega[:,:,:,None],
            taus)
        # (B, S, E, M) -> # (B, S, M, E)
        cell_tau = cell_tau.transpose(2, 3)
        # (B, S, M, E) -> (B, S, K, M)
        intens_at_samples = self.intensity_layer(cell_tau).transpose(2,3)
        # In case of different long sequences, make all equal length and pad with 0, 
        intens_at_samples = nn.utils.rnn.pad_sequence(
            intens_at_samples, padding_value=0.0, batch_first=True)
        # Sum intensities over event category type (B, S, K, M) -> (B, S, M)
        total_intens_samples = intens_at_samples.sum(dim=2)  # shape batch * N * MC
        # Take mean over monte carlo samples of intensity per event and 
        # pointwise multiply with inter-event-times -> (B, S)
        partial_integrals = dt_seq * total_intens_samples.mean(dim=2)

        # Get integrated value for each batch (B, S) -> (B, )
        integral_ = partial_integrals.sum(dim=1)

        res = torch.sum(- log_sum + integral_)
        return res#, log_intensities, log_intensities_ret, partial_integrals


    def read_predict(self, seq_times, seq_types, seq_lengths, pad, device,
                     hmax = 40, n_samples=1000, plot = False, print_info = False):
        """
        Read an event sequence and predict the next event time and type.

        Args:
            seq_times: # start from 0
            seq_types:
            seq_lengths:
            hmax:
            plot:
            print_info:

        Returns:

        """

        length = seq_lengths.item()  # exclude the first added event

        ## remove the first added event
        dt_seq = seq_times[1:] - seq_times[:-1]
        last_t = seq_times[length - 1]
        next_t = seq_times[length]

        dt_seq_valid = dt_seq[:length]  # exclude the last timestamp
        dt_seq_used = dt_seq_valid[:length-1]  # exclude the last timestamp
        next_dt = dt_seq_valid[length - 1]

        seq_types_valid = seq_types[1:length + 1]  # include the first added event
        from train_functions.train_sahp import MaskBatch
        last_type = seq_types[length-1]
        next_type = seq_types[length]
        if next_type == self.process_dim:
            print('Error: wrong next event type')
        seq_types_used = seq_types_valid[:-1]
        seq_types_valid_masked = MaskBatch(seq_types_used[None, :], pad, device)
        seq_types_used_mask = seq_types_valid_masked.src_mask


        with torch.no_grad():
            self.forward(dt_seq_used, seq_types_used, seq_types_used_mask)

            if self.omega.shape[1] == 0:  # only one element
                estimate_dt, next_dt, error_dt, next_type, estimate_type = 0,0,0,0,0
                return estimate_dt, next_dt, error_dt, next_type, estimate_type

            elif self.omega.shape[1] == 1: # only one element
                converge_point = torch.squeeze(self.converge_point)[None, :]
                start_point = torch.squeeze(self.start_point)[None,:]
                omega = torch.squeeze(self.omega)[None, :]
            else:
                converge_point = torch.squeeze(self.converge_point)[-1, :]
                start_point = torch.squeeze(self.start_point)[-1, :]
                omega = torch.squeeze(self.omega)[-1, :]

            dt_vals = torch.linspace(0, hmax, n_samples + 1).to(device)
            h_t_vals = self.state_decay(converge_point,
                                        start_point,
                                        omega,
                                        dt_vals[:, None])
            if print_info:
                print("last event: time {:.3f} type {:.3f}"
                      .format(last_t.item(), last_type.item()))
                print("next event: time {:.3f} type {:.3f}, in {:.3f}"
                      .format(next_t.item(), next_type.item(), next_dt.item()))

            return predict_from_hidden(self, h_t_vals, dt_vals, next_dt, next_type,
                                            plot, hmax, n_samples, print_info)


    def plot_estimated_intensity(self,timestamps, n_points=10000, plot_nodes=None,
                                 t_min=None, t_max=None,
                                 intensity_track_step=None, max_jumps=None,
                                 show=True, ax=None, qqplot=None):
        from simulation.simulate_hawkes import fuse_node_times
        event_timestamps, event_types = fuse_node_times(timestamps)

        event_timestamps = torch.from_numpy(event_timestamps)
        seq_times = torch.cat((torch.zeros_like(event_timestamps[:1]), event_timestamps),
                              dim=0).float()  # add 0 to the sequence beginning
        dt_seq = seq_times[1:] - seq_times[:-1]

        seq_types = torch.from_numpy(event_types)
        seq_types = seq_types.long()# convert from floattensor to longtensor

        intens_at_evs_lst = []
        sample_times = np.linspace(t_min, t_max, n_points)
        for i in range(self.process_dim):
            intens_at_samples, intens_at_evs = self.intensity_per_type(seq_types, dt_seq, sample_times, timestamps[i], type=i)
            intens_at_evs_lst.append(intens_at_samples)
            if qqplot is None:
                self._plot_tick_intensity(timestamps[i], sample_times, intens_at_samples,intens_at_evs,
                                          ax[i], i, n_points)
        if qqplot is not None:
            return intens_at_evs_lst

    def intensity_per_type(self, seq_types, dt_seq, sample_times, timestamps, type):
        from train_functions.train_sahp import MaskBatch

        intens_at_samples = []
        with torch.no_grad():

            onetype_length = timestamps.size
            alltype_length = len(seq_types)

            type_idx = np.arange(alltype_length)[seq_types == type]

            event_types_masked = MaskBatch(seq_types[None, :], pad=self.process_dim, device='cpu')
            event_types_mask = event_types_masked.src_mask

            self.forward(dt_seq, seq_types, event_types_mask)
            converge_point = torch.squeeze(self.converge_point)
            start_point = torch.squeeze(self.start_point)
            omega = torch.squeeze(self.omega)

            cell_t = self.state_decay(converge_point,
                                      start_point,
                                      omega,
                                      dt_seq[:, None])#

            intens_at_evs = torch.squeeze(self.intensity_layer(cell_t)).numpy()
            intens_at_evs = intens_at_evs[type_idx, type]


            event_idx = -1
            for t_time in sample_times:
                if t_time < timestamps[0]:
                    intens_at_samples.append(0)#np.zeros(self.process_dim)
                    continue

                if event_idx < onetype_length - 1 and t_time >= timestamps[event_idx + 1]:
                    event_idx += 1
                    # print(omega)

                aaa=dt_seq[:event_idx+1]
                bbb=seq_types[:event_idx+1]

                event_types_masked = MaskBatch(bbb[None, :], pad=self.process_dim, device='cpu')
                event_types_mask = event_types_masked.src_mask

                self.forward(aaa, bbb, event_types_mask)

                converge_point = torch.squeeze(self.converge_point)
                start_point = torch.squeeze(self.start_point)
                omega = torch.squeeze(self.omega)

                if omega.ndim == 2:
                    omega = omega[-1,:]
                    converge_point = converge_point [-1,:]
                    start_point = start_point[-1,:]
                cell_t = self.state_decay(converge_point,
                                          start_point,
                                          omega,
                                          t_time - timestamps[event_idx])#

                xxx = self.intensity_layer(cell_t).numpy()
                intens_at_samples.append(xxx[type])


            return intens_at_samples, intens_at_evs

    def _plot_tick_intensity(self, timestamps_i, sample_times, intensity_i, intens_at_evs,
                             ax, label, n_points):#
        x_intensity = np.linspace(sample_times.min(), sample_times.max(), n_points)
        y_intensity = intensity_i
        ax.plot(x_intensity, y_intensity)

        ax.set_title(label)


class SAHPGen(SeqGenerator):
    # sequence generator for the SAHP model

    def __init__(self,model, record_intensity = True):
        super(SAHPGen, self).__init__(model, record_intensity)
        self.lbda_ub = []

    def _restart_sequence(self):
        super(SAHPGen, self)._restart_sequence()
        self.lbda_ub = []
