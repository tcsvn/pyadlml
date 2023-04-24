from torch import nn
from pyadlml.model.nn import Tanh, Embedding, FlattenConsecutive
from torch.nn import Linear, BatchNorm1d
import torch 
import torch.nn.functional as F
import numpy as np
from pyadlml.model.nn.util import CausalConv1d, DilatedConv1d
import math

def print_in_out(layer, name=None):
    def wrapper(x):
        i = list(x.shape)
        x = layer(x)
        j = list(x.shape)
        name = layer.__class__.__name__ if name is None else name
        print(f'{name} {i} -> {j}')
        return x
    return wrapper


class DilatedModel(nn.Module):

    def __init__(self, n_features, n_classes, seq_length, n_emb=3, n_hidden=100, dilation_factor=2):
        """
        """
        super().__init__()

        # Calculate 
        nr_layers = math.log(seq_length, dilation_factor)
        assert nr_layers % 1 == 0, 'Sequence length has to be a exponential of dilation factor to form usable layers'

        class CausalBlock(nn.Module):
            def __init__(self, dilation_factor, lin_in_dim, n_hidden):
                super().__init__()
                self.d = dilation_factor
                self.flc = FlattenConsecutive(dilation_factor)      # B, T, E     -> B, T/d, E*d
                self.lin = Linear(lin_in_dim, n_hidden, bias=False) # B, T/d, E*d -> B, T/d, H
                self.bn1d = BatchNorm1d(n_hidden)                   # B, T/d, H   -> B, T/d, H
                self.tanh = Tanh()                                  # B, T/d, H   -> B, T/d, H

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                B, T, C = x.shape

                x = self.flc(x)
                T = T//self.d

                x = self.lin(x)
                C = self.lin.out_features

                x = x.view(B, C, T)
                x = self.bn1d(x)
                x = x.view(B, T, C)
                self.tanh(x)
                return x

        layers = []
        for i in range(int(nr_layers)):
            if i == 0:
                in_lin_dim = n_features*dilation_factor
            else:
                in_lin_dim = n_hidden*dilation_factor
            layers.append(CausalBlock(dilation_factor, in_lin_dim, n_hidden))

        self.layers = nn.ModuleList(layers)
        self.ff = nn.Linear(n_hidden, n_classes)


    def compute_loss(self, logits_cls, y_cls, weight):
        loss_cls = F.cross_entropy(logits_cls, y_cls, weight=weight) 
        # Refactor
        praefix = 'train' if self.training else 'val'

        return loss_cls

    @classmethod
    def gen_nr_layer_dil_pairs(cls, dilation_factor, nr_lengths=10):
        return [(i, dilation_factor**i) for i in range(1, nr_lengths+1)]


    def forward(self, x):
        """
        """
        B, T, F = x.shape

        # (B,T,F) -> ()
        for l in self.layers:
            x = l(x)

        x = self.ff(x)

        return x.squeeze()


class ResidualBlock(torch.nn.Module):
    def __init__(self, res_channels, skip_channels, dilation):
        """
        Residual block
        :param res_channels: number of residual channel for input, output
        :param skip_channels: number of skip channel for output
        :param dilation:
        """
        super(ResidualBlock, self).__init__()

        self.dilate_filter = DilatedConv1d(
            res_channels, res_channels,
            kernel_size=2, stride=1,  
            dilation=dilation,
            bias=False
        )  
        
        self.dilate_gate = DilatedConv1d(
            res_channels, res_channels,
            kernel_size=2, stride=1,  
            dilation=dilation,
            bias=False
        )

        # 1x1 convolutions
        self.conv_res = torch.nn.Conv1d(res_channels, res_channels, 1)
        self.conv_skip = torch.nn.Conv1d(res_channels, skip_channels, 1)

    def forward(self, x, skip_size):
        """
        Parameters
        ----------
        x : torch.tensor of shape (B, R, T) 
            where B is batch size, R is #residual dimensions and T is the sequence length

        Returns
        -------
        torch.tensor of shape (B, S, T)
            where B is batch size, S is the #skip dimensions and T is the sequence length

        """
        B, R, T = x.shape

        # Follows eq. 2 from paper rather than diagram
        x_sigm = self.dilate_filter(x)  # (B, R, T) -> (B, R, T)
        x_tanh = self.dilate_gate(x)    # (B, R, T) -> (B, R, T)

        # PixelCNN gate
        gated_tanh = F.tanh(x_tanh)
        gated_sigmoid = F.sigmoid(x_sigm)
        gated = gated_tanh * gated_sigmoid

        # Residual connection
        output = self.conv_res(gated)

        x = x + output

        # Skip connection
        skip = self.conv_skip(gated)

        return output, skip


class WaveNetModel(torch.nn.Module):
    """
    Inspired by:
        https://github.com/golbin/WaveNet/blob/master/wavenet/networks.py 
    and
        https://github.com/vincentherrmann/pytorch-wavenet/blob/26ba28989edcf8688f6216057aafda07601ff07e/wavenet_model.py#L144

    
    """
    def __init__(self, n_layers, n_blocks, in_channels, res_channels, skip_channels, 
                       end_channels, n_features, n_classes):
        """
        Stack residual blocks by layer and stack size

        Parameters
        ----------
        :param layer_size: integer, 10 = layer[dilation=1, dilation=2, 4, 8, 16, 32, 64, 128, 256, 512]
        :param stack_size: integer, 5 = stack[layer1, layer2, layer3, layer4, layer5]
        :param in_channels: number of channels for input data. skip channel is same as input channel
        :param res_channels: number of residual channel for input, output
        :return:
        """
        super(WaveNetModel, self).__init__()
        self.layer_size = n_layers
        self.stack_size = n_blocks
        self.res_channels = res_channels
        self.skip_channels = skip_channels
        self.in_channels = in_channels

        self.receptive_fields =  self.calc_receptive_fields(n_layers, n_blocks)

        # Build causal conv
        # Padding=1 for same size(length) between input and output for causal convolution
        self.causal = CausalConv1d(in_channels, res_channels, kernel_size=2, stride=1, bias=False)

        # Build Residual layers
        # For s=3, l=4 -> [1,2,4,8,  1,2,4,8,  1,2,4,8]
        dilations = [ 2**l for s in range(0, self.stack_size) for l in range(0, self.layer_size)]
        
        self.res_layers = torch.nn.ModuleList([
                            ResidualBlock(res_channels, skip_channels, dilation)
                            for dilation in dilations
        ])

        self.densnet = torch.nn.Sequential(*[
            torch.nn.ReLU(),
            torch.nn.Conv1d(skip_channels, end_channels, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(end_channels, n_features, kernel_size=1),
            torch.nn.Softmax(dim=1),
        ])

        # No downsampling for low timestep sizes
        self.classif_net = torch.nn.Sequential(*[
            torch.nn.ReLU(),
            # (B,S,T) -> (B, 100, T)
            torch.nn.Conv1d(skip_channels, 100, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.Conv1d(100, 50, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Conv1d(50, 20, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(1880, n_classes)
            #torch.nn.Linear(20*10, n_classes)
        ])

    @staticmethod
    def calc_receptive_fields(layer_size, stack_size):
        
        layers = [2 ** i for i in range(0, layer_size)] * stack_size
        num_receptive_fields = np.sum(layers)

        return int(num_receptive_fields)

    def calc_output_size(self, x):
        output_size = int(x.size(2)) - self.receptive_fields

        self.check_input_size(x, output_size)

    def forward(self, x):
        """
        The size of timestep(3rd dimention) has to be bigger than receptive fields
        :param x: Tensor[batch, timestep, channels]
        :return: Tensor[batch, timestep, channels]

        Parameters
        ----------
        x: torch.tensor
            tensor with shape (B, T, C)

        Returns
        -------
        torch.tensor
            Shape
        """
        B, T, C, R, S = [*x.shape, self.res_channels, self.skip_channels]

        # (B,T,C) -> (B,C,T)
        # Move T to last dimension since most torch operations assume seq length as such
        output = x.transpose(1, 2)   

        # (B,C,T) -> (B,R,T)
        output = self.causal(output)

        skip_stream = torch.zeros((B, S, T), dtype=output.dtype, device=output.device)
        for res_block in self.res_layers:
            # (B,R,T) -> (B,R,T), (B,S,T)
            output, skip_i = res_block(output, S)
            skip_stream = skip_stream + skip_i

        # (B,S,T) -> (B,C,T)
        output = self.densnet(skip_stream)

        logits_next_dev = output.swapaxes(1,2)

        # Take last ouput after applying all res_blocks
        logits_cls = self.classif_net(skip_i)


        return logits_next_dev, logits_cls


    def compute_loss(self, logits_next_dev, y_next_dev, logits_cls, y_cls, weight, rep='many-to-many'):
        B, T, C_d, C_c = [*logits_next_dev.shape, y_cls.shape[-1]]
        # TODO critical, build data loader for next item prediction
        if rep == 'many-to-one':
            # Take last symbol as next dev prediction many-to-one 
            # (B,T,C_d) -> (B,C_d)
            logits_next_dev = logits_next_dev[:,:-1,:]
            # (B,T) -> (B,)
            y_next_dev = y_next_dev[:, -1]
            
        if rep == 'many-to-many':
            # (B,T,C_d) -> (B*T, C_d)
            logits_next_dev = logits_next_dev.reshape(B*T, C_d)
            # (B,T) -> (B*T)
            y_next_dev = y_next_dev.view(B*T)

        

        loss_cls = F.cross_entropy(logits_cls, y_cls, weight=weight) 
        loss_nd = F.cross_entropy(logits_next_dev, y_next_dev)

        loss = loss_cls + loss_nd

        # Refactor
        praefix = 'train' if self.training else 'val'
        import mlflow
        mlflow.log_metric(f'{praefix}_loss_cls', loss_cls.item())
        mlflow.log_metric(f'{praefix}_loss_next_dev', loss_nd.item())

        return loss
