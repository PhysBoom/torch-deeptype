from typing import Sequence
from torch import Tensor, nn
import torch
from deeptype_model import DeeptypeModel

class SparsityLoss(nn.Module):
    def __init__(self):
        super(SparsityLoss, self).__init__()

    def forward(self, model: DeeptypeModel):
        w = model.get_input_layer_weights()
        # Note - The original paper uses row-wise losses, but this
        # is incorrect if we want to push individual input weights to as close
        # to 0 as possible since the weights for an input will be in a column.
        return torch.norm(w, p=2, dim = 0).sum()
    

class ClusterRepresentationLoss(nn.Module):
    """
    Loss detecting difference between representation layer and KNN cluster centers.
    Computes the average Euclidean distance between each hidden representation
    and its corresponding cluster center.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        model: DeeptypeModel,
        inputs: Tensor,
        cluster_centers: Sequence[Tensor]
    ) -> Tensor:
        batch_size = inputs.size(0)
        if len(cluster_centers) != batch_size:
            raise ValueError(
                f"Expected {batch_size} cluster centers, "
                f"but got {len(cluster_centers)}."
            )

        # get hidden reps: (batch_size, hidden_dim)
        reps = model.get_hidden_representations(inputs)

        # stack centers into (batch_size, hidden_dim)
        centers = torch.stack(cluster_centers, dim=0)
        centers = centers.to(reps.device).type_as(reps)

        # compute per-sample Euclidean distances
        dists = torch.norm(reps - centers, p=2, dim=1)

        # return mean distance
        return dists.mean()