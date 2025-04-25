# torch-deeptype

PyTorch implementation of DeepType.

## Installation

Run `pip install torch-deeptype`

## Usage

Usage
After installing (pip install torch-deeptype), follow these steps:

1. Define your model
Create a DeeptypeModel subclass that implements:

`forward(self, x: Tensor) -> Tensor`
`get_input_layer_weights(self) -> Tensor`
`get_hidden_representations(self, x: Tensor) -> Tensor`

**Tip:** Have forward() call get_hidden_representations() to avoid duplicating the hidden-layer code.

```python
import torch
import torch.nn as nn
from torch_deeptype import DeeptypeModel

class MyNet(DeeptypeModel):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_layer   = nn.Linear(input_dim, hidden_dim)
        self.h1            = nn.Linear(hidden_dim, hidden_dim)
        self.cluster_layer = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output_layer  = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Notice how forward() gets the hidden representations
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

2. Prepare your data
Wrap your tensors in a TensorDataset and DataLoader as usual:

```python
from torch.utils.data import TensorDataset, DataLoader

# Example with random data:
X = torch.randn(1000, 20)         # 1000 samples, 20 features
y = torch.randint(0, 5, (1000,))  # 5 classes

dataset      = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
```

3. Instantiate the trainer
Use DeeptypeTrainer to set up both phases of DeepType training:

```python
from torch_deeptype import DeeptypeTrainer

trainer = DeeptypeTrainer(
    model           = MyNet(input_dim=20, hidden_dim=64, output_dim=5),
    train_loader    = train_loader,
    primary_loss_fn = nn.CrossEntropyLoss(),
    num_clusters    = 8,       # K in KMeans
    sparsity_weight = 0.01,    # α for L₂ sparsity on input weights
    cluster_weight  = 0.5,     # β for cluster‐rep loss
    verbose         = True     # print per-epoch loss summaries
)
```

4. Run training
Call trainer.train(...) to execute the Deeptype training

```python
trainer.train(
    main_epochs           = 15,     # epochs for joint phase
    main_lr               = 1e-4,   # LR for joint phase
    pretrain_epochs       = 10,     # epochs for pretrain phase
    pretrain_lr           = 1e-3,   # LR for pretrain (defaults to main_lr if None)
    train_steps_per_batch = 8       # inner updates per batch in joint phase
)
```

With `verbose=True`, you’ll see three loss components logged each epoch:
- Primary (classification/regression loss)
- Sparsity (input-weight L₂ penalty)
- Cluster (hidden-representation vs. KMeans centers)

5. Extract clusters and important inputs

After training, you can inspect:
- KMeans clusters over your dataset’s hidden representations
- Input‐feature importances via the L₂‐norm of each input weight column

```python
from torch.utils.data import TensorDataset

# 1) Prepare the same dataset you trained on
dataset = TensorDataset(X, y)

# 2) Compute clusters
#    Returns:
#      - `centroids`: Tensor[num_clusters, hidden_dim]
#      - `labels`:    np.ndarray[N] of cluster assignments
centroids, labels = trainer.get_clusters(dataset)

print("Centroids shape:", centroids.shape)
print("Cluster assignments for first 10 samples:", labels[:10])


# 3) Compute input‐feature importance (on your model)
#    importance[i] = || W[:, i] ||₂ for first‐layer weights W
importances = trainer.model.get_input_importance()
print("Importances:", importances)

# 4) Get features sorted by importance
#    returns a Tensor of feature indices, most important first
sorted_idx = trainer.model.get_sorted_input_indices()
print("Top 5 features by importance:", sorted_idx[:5].tolist())
```

That’s all you need to get DeepType running end-to-end!

If you're a more advanced user, you can also use the `SparsityLoss` and `ClusterRepresentationLoss` directly.

## Acknowledgements

This implementation is based on Runpu Chen's original implementation [here](https://github.com/runpuchen/DeepType.git). The original paper that introduced DeepType can be found [here](https://pubmed.ncbi.nlm.nih.gov/31603461/).