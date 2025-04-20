import unittest
import math
import torch
from losses import SparsityLoss  # your SparsityLoss must now use dim=0 internally
from torch import nn

class DummyModel(nn.Module):
    def __init__(self, weight_tensor: torch.Tensor):
        """
        weight_tensor: shape (hidden_dim, input_dim)
        """
        super().__init__()
        hidden_dim, input_dim = weight_tensor.shape
        self.input_layer = nn.Linear(input_dim, hidden_dim, bias=False)
        # overwrite default weight
        with torch.no_grad():
            self.input_layer.weight.copy_(weight_tensor)
        self.input_layer.weight.requires_grad_(True)

class TestSparsityLoss(unittest.TestCase):

    def test_forward_known_values(self):
        """Column‑wise norms against hand‑computed sums."""
        cases = [
            # all zeros → loss 0
            (torch.zeros(3, 5), 0.0),
            # weight = [[3,0],[4,0]] → col norms = [√(3²+4²)=5, 0] → sum=5
            (torch.tensor([[3.0, 0.0],
                           [4.0, 0.0]]), 5.0),
            # weight = [[1,2],[3,4]] → col norms = [√(1+9)=√10, √(4+16)=√20] → sum=√10+√20
            (torch.tensor([[1.0, 2.0],
                           [3.0, 4.0]]), math.sqrt(10) + math.sqrt(20)),
        ]
        loss_fn = SparsityLoss()
        for weight, expected in cases:
            with self.subTest(weight=weight, expected=expected):
                model = DummyModel(weight)
                loss = loss_fn(model)
                self.assertTrue(
                    torch.allclose(loss, torch.tensor(expected), atol=1e-6),
                    f"Expected {expected}, got {loss.item()}"
                )

    def test_gradient_flow(self):
        """Gradients: ∂/∂w_{i,j} = w_{i,j} / ‖column_j‖₂."""
        # single row [[3,4]] → col norms = [3,4] → loss = 3 + 4 = 7
        weight = torch.tensor([[3.0, 4.0]], requires_grad=True)
        model = DummyModel(weight)
        loss_fn = SparsityLoss()

        loss = loss_fn(model)
        self.assertAlmostEqual(loss.item(), 7.0, places=6)

        loss.backward()
        grads = model.input_layer.weight.grad
        # grads = [3/3, 4/4] = [1,1]
        expected_grads = torch.tensor([[1.0, 1.0]])
        self.assertTrue(
            torch.allclose(grads, expected_grads, atol=1e-6),
            f"Expected grads {expected_grads}, got {grads}"
        )

    def test_independent_models(self):
        """Two models with different weights get independent column‑wise losses."""
        w1 = torch.tensor([[1.0, 2.0],
                           [0.0, 0.0]])
        w2 = torch.tensor([[0.0, 3.0],
                           [4.0, 0.0]])
        m1, m2 = DummyModel(w1), DummyModel(w2)
        loss_fn = SparsityLoss()

        # m1: col norms = [1,2] → sum = 3
        l1 = loss_fn(m1)
        # m2: col norms = [4,3] → sum = 7
        l2 = loss_fn(m2)

        self.assertAlmostEqual(l1.item(), 3.0, places=6)
        self.assertAlmostEqual(l2.item(), 7.0, places=6)
        
    def test_training_reduces_weights_to_zero(self):
        """Train only on column‑wise sparsity loss for 100 epochs → all weights ≈ 0."""
        torch.manual_seed(0)
        init_w = torch.randn(4, 6)
        model = DummyModel(init_w.clone())
        loss_fn = SparsityLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        # initial column‑wise ℓ₂,₁ norm
        initial_norm = torch.norm(model.input_layer.weight.data, p=2, dim=0).sum().item()

        for _ in range(100):
            optimizer.zero_grad()
            loss = loss_fn(model)
            loss.backward()
            optimizer.step()

        final_norm = torch.norm(model.input_layer.weight.data, p=2, dim=0).sum().item()

        # Should drop to <10% of initial
        self.assertLess(final_norm, initial_norm * 0.1,
                        f"Final norm {final_norm:.6f} not <10% of initial {initial_norm:.6f}")

if __name__ == "__main__":
    unittest.main()
