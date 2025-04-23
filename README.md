# torch-deeptype

PyTorch implementation of DeepType.

## Installation

Run `pip install torch-deeptype`

## Usage

To start, create a model that extends the DeeptypeModel class. These models need to have three methods:

1) `forward(self, x: Tensor) -> Tensor` - Run the prediction (as is normally required in Pytorch)
2) `get_input_layer_weights(self) -> Tensor` - Return the raw weight tensor of the first (input) layer. 
3) `get_hidden_representations(self, x: Tensor) -> Tensor` - Given an input batch `x`, run everything up to—but excluding—the final output layer, and return those activations (i.e. the second-to-last layer).

Note that, as per DRY principles, it's typically best to have `forward()` call `get_hidden_representations()` instead of duplicating code.

An example is shown below:

```{python}
import torch
import torch.nn as nn
from torch_deeptype import DeeptypeModel

class MyNet(DeeptypeModel):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_layer   = nn.Linear(input_dim, hidden_dim)
        self.h1            = nn.Linear(hidden_dim, hidden_dim)
        self.cluster_layer = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output_layer  = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Re-use get_hidden_representations for the penultimate activations
        hidden = self.get_hidden_representations(x)
        return self.output_layer(hidden)

    def get_input_layer_weights(self) -> torch.Tensor:
        return self.input_layer.weight

    def get_hidden_representations(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.input_layer(x))
        x = torch.relu(self.h1(x))
        x = torch.relu(self.cluster_layer(x))
        return x
```