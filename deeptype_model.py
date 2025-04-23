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
    
    def train(self,)
