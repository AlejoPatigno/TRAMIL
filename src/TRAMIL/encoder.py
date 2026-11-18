"""Trainable acoustic encoding H = ReLU(X W_x + b_x), 88 -> 32 (Ec. 13, 14)."""
from torch import nn

__provenance__ = {
    "revision": "research-concept-residual-semantic-prior-bipartite-mil-r06.md",
    "sections": ["6"],
    "equations": [13, 14],
    "invariants": ["encoder_is_a_nonnegative_32d_relu_projection"],
}


class Encoder(nn.Module):
    def __init__(self, d_in=88, h=32):
        super().__init__()
        self.fc = nn.Linear(d_in, h)

    def forward(self, X):
        return nn.functional.relu(self.fc(X))
