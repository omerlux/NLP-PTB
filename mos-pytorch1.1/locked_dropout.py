import torch
import torch.nn as nn
import numpy as np
import model

class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout, self).__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        # mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = m.div_(1 - dropout).detach()     # TODO: ask cameron why is this for? the m is already 1's and 0's...
        mask = mask.expand_as(x)
        return mask * x


class MyLockedDropout(nn.Module):
    def __init__(self):
        super(MyLockedDropout, self).__init__()

    def forward(self, layer, old_mask=None, dropout=0.5):
        # if there is no dropout OR (we aren't in training AND not in MonteCarlo evaluation):
        if dropout == 0 or not self.training:
            return layer    # no implementation of dropout
        else:
            # building a mask
            _seq_len = layer.data.shape[0]
            batch_size = layer.data.shape[1]
            neurons = layer.data.shape[2]
            probability = 1 - dropout
            # Tensor.data.new = duplicate type and device of the tensor
            mask = layer.data.new(np.random.binomial(1, p=probability, size=(1, batch_size, neurons)))
            # Tensor.expand_as(x) = will generate many mask as the size of x first element
            mask = mask.expand_as(layer)
            return layer * mask     # dropping-out layer's neurons

    def getMask(self, layer, dropout=0.5):
        _seq_len = layer.data.shape[0]
        batch_size = layer.data.shape[1]
        neurons = layer.data.shape[2]
        # if there is no dropout OR (we aren't in training AND not in MonteCarlo evaluation):
        if dropout != 0 or not self.training:
            mask_ones = layer.data.new(np.ones(size=(1, batch_size, neurons)))
            return mask_ones    # returns impotent mask (only 1's)
        else:
            # building a mask
            probability = 1 - dropout
            # Tensor.data.new = duplicate type and device of the tensor
            mask = layer.data.new(np.random.binomial(1, p=probability, size=(1, batch_size, neurons)))
            # Tensor.expand_as(x) = will generate many mask as the size of x first element
            mask = mask.expand_as(layer)
            return mask             # returning only the mask!