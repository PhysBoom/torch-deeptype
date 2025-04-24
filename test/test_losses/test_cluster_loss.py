import unittest
import math
import torch
import numpy as np
from torch import nn
from torch_deeptype.deeptype_model import DeeptypeModel
from torch_deeptype.losses import ClusterRepresentationLoss
from sklearn.cluster import KMeans

class DummyIdentityModel(DeeptypeModel):
    """A DeeptypeModel whose hidden representations are exactly the inputs."""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def get_input_layer_weights(self) -> torch.Tensor:
        return torch.empty(0)

    def get_hidden_representations(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
class SmallNet(DeeptypeModel):
    """
    A small trainable network: 
    input 3-d → hidden 10-d → representation 3-d.
    """
    def __init__(self, input_dim=3, hidden_dim=10, rep_dim=3):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.rep_layer   = nn.Linear(hidden_dim, rep_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # not used by ClusterRepresentationLoss itself
        h = torch.relu(self.input_layer(x))
        return self.rep_layer(h)

    def get_input_layer_weights(self) -> torch.Tensor:
        return self.input_layer.weight

    def get_hidden_representations(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.input_layer(x))
        return self.rep_layer(h)

class TestClusterRepresentationLoss(unittest.TestCase):
    def setUp(self):
        self.loss_fn = ClusterRepresentationLoss()
        self.model = DummyIdentityModel()

    def test_forward_known_values(self):
        """Euclidean distances between reps and centers match hand-computed values."""
        cases = [
            # zeros → loss 0
            (
                torch.zeros(2, 3),
                [torch.zeros(3), torch.zeros(3)],
                0.0
            ),
            # single sample [[3,4]] vs [0,0] → distance = 5
            (
                torch.tensor([[3.0, 4.0]]),
                [torch.tensor([0.0, 0.0])],
                5.0
            ),
            # exact match → loss 0
            (
                torch.tensor([[1.0, 2.0],
                              [3.0, 4.0]]),
                [torch.tensor([1.0, 2.0]),
                 torch.tensor([3.0, 4.0])],
                0.0
            ),
            # mix: [[1,2],[3,4]] vs [[1,0],[0,4]]
            # d1 = sqrt((1-1)^2 + (2-0)^2) = 2
            # d2 = sqrt((3-0)^2 + (4-4)^2) = 3
            # mean = (2 + 3)/2 = 2.5
            (
                torch.tensor([[1.0, 2.0],
                              [3.0, 4.0]]),
                [torch.tensor([1.0, 0.0]),
                 torch.tensor([0.0, 4.0])],
                2.5
            ),
        ]
        for inputs, centers, expected in cases:
            with self.subTest(inputs=inputs, centers=centers, expected=expected):
                loss = self.loss_fn(self.model, inputs, centers)
                self.assertAlmostEqual(
                    loss.item(), expected, places=6,
                    msg=f"Expected loss {expected}, got {loss.item()}"
                )

    def test_mismatch_centers_raises(self):
        """Passing wrong number of centers raises ValueError."""
        inputs = torch.randn(5, 4)
        centers = [torch.randn(4)] * 4  # should be length 5
        with self.assertRaises(ValueError):
            _ = self.loss_fn(self.model, inputs, centers)

    def test_gradient_flow_wrt_inputs(self):
        """Gradient wrt inputs is (x - c) / ‖x - c‖."""
        inp = torch.tensor([[1.0, 3.0]], requires_grad=True)
        centers = [torch.tensor([0.0, 0.0])]
        loss = self.loss_fn(self.model, inp, centers)
        # distance = sqrt(1^2 + 3^2) = sqrt(10)
        self.assertAlmostEqual(loss.item(), math.sqrt(10), places=6)
        loss.backward()
        # grad = (inp - 0) / sqrt(10)
        expected_grad = torch.tensor([[1.0, 3.0]]) / math.sqrt(10)
        self.assertTrue(
            torch.allclose(inp.grad, expected_grad, atol=1e-6),
            f"Expected grad {expected_grad}, got {inp.grad}"
        )

    def test_training_with_kmeans_clusters(self):
        """
        Real-life example: cluster 3D points, then train a small MLP so its
        representations move closer to their assigned centers.
        """
        torch.manual_seed(0)
        np.random.seed(0)

        # generate 20 random 3-D points
        data_np = np.random.randn(20, 3)
        kmeans = KMeans(n_clusters=3, random_state=0).fit(data_np)
        labels = kmeans.labels_
        centers_np = kmeans.cluster_centers_

        inputs = torch.tensor(data_np, dtype=torch.float32)
        cluster_centers = [
            torch.tensor(centers_np[label], dtype=torch.float32)
            for label in labels
        ]

        # use the small trainable network
        model = SmallNet(input_dim=3, hidden_dim=10, rep_dim=3)
        loss_fn = ClusterRepresentationLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        # compute initial per-sample distances
        with torch.no_grad():
            reps_init = model.get_hidden_representations(inputs)
        centers_tensor = torch.stack(cluster_centers, dim=0)
        initial_dists = torch.norm(reps_init - centers_tensor, p=2, dim=1)

        initial_loss = initial_dists.mean().item()

        # train only on the CRL for 50 steps
        for _ in range(50):
            optimizer.zero_grad()
            loss = loss_fn(model, inputs, cluster_centers)
            loss.backward()
            optimizer.step()

        # compute final per-sample distances
        with torch.no_grad():
            reps_final = model.get_hidden_representations(inputs)
        final_dists = torch.norm(reps_final - centers_tensor, p=2, dim=1)
        final_loss = final_dists.mean().item()

        # overall loss should decrease
        self.assertLess(
            final_loss, initial_loss,
            f"Final loss {final_loss:.6f} not less than initial {initial_loss:.6f}"
        )

        # all of our points should be closer to their cluster center than before
        # Note: this isn't necessarily true provably, but usually should be, so the test is fine for now.
        self.assertTrue(
            torch.all(final_dists <= initial_dists + 1e-6),
            f"Some points did not get closer:\ninitial distances {initial_dists}\nfinal distances   {final_dists}"
        )

if __name__ == "__main__":
    unittest.main()