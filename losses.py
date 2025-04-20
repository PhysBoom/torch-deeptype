from torch import nn
import torch

class SparsityLoss(nn.Module):
    def __init__(self):
        super(SparsityLoss, self).__init__()

    def forward(self, model):
        w = model.input_layer.weight
        # Note - The original paper uses row-wise losses, but this
        # is incorrect if we want to push individual input weights to as close
        # to 0 as possible since the weights for an input will be in a column.
        return torch.norm(w, p=2, dim = 0).sum()
