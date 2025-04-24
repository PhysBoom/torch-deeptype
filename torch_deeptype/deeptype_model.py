import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class DeeptypeModel(nn.Module, ABC):
    """
    Base class for DeepType-style models.
    Subclasses must implement:
      - forward: the usual nn.Module forward pass
      - get_input_layer_weights: return the input-layer weights tensor
      - get_hidden_representations: return the penultimate layer activations
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the model’s output.
        """
        pass

    @abstractmethod
    def get_input_layer_weights(self) -> torch.Tensor:
        """
        Return the weight matrix of the first (input) layer.
        """
        pass

    @abstractmethod
    def get_hidden_representations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given inputs x, compute and return the activations
        of the second-to-last (hidden) layer.
        """
        pass
    
    def get_input_importance(self) -> torch.Tensor:
        """
        Compute an “importance score” for each input feature, defined as
        the L₂-norm of its column in the first‐layer weight matrix.

        Returns:
            importance (Tensor[input_dim]):
                importance[i] = || W[:, i] ||₂
        """
        # fetch the input‐layer weights (shape: [hidden_dim, input_dim])
        w = self.get_input_layer_weights()
        # compute the column‐wise L2 norm
        importance = torch.norm(w, p=2, dim=0)
        return importance

    def get_sorted_input_indices(self) -> torch.Tensor:
        """
        Return the indices of the input features, sorted by descending importance.

        Internally calls `get_input_importance()` and then argsorts.

        Returns:
            sorted_idx (Tensor[input_dim]):
                A permutation of [0..input_dim-1], with the most important
                feature first.
        """
        importance = self.get_input_importance()
        # argsort with descending=True
        sorted_idx = torch.argsort(importance, descending=True)
        return sorted_idx
