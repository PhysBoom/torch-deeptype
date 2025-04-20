# torch-deeptype

PyTorch implementation of DeepType.

## Usage

To start, create a model like you normally would in Pytorch, e.g.

```{python}
class MyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Note: It is imporant that your model has an input_layer
        # and cluster_layer parameter.
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.h1 = nn.Linear(512, 256)
        self.cluster_layer = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = torch.relu(self.hidden(x))
        return self.output_layer(x)
```

As the notes say, ensure that your model has an `input_layer` and `output_layer` defined - this is used to attach the losses.
