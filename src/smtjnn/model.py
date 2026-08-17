"""Stochastic binary MLP (p-bit style: stochastic at inference, not just training).

Each hidden unit computes pre-activation I, fires s = +1 with probability
p = (1 + tanh(I)) / 2 and s = -1 otherwise, every time it is queried.
Training uses ideal i.i.d. uniform randomness and a straight-through
estimator (gradient of the mean activation tanh(I)).

At inference the Bernoulli draw is delegated to a pluggable
`source.sample(p, layer)` so imperfect physical randomness (LFSR, sMTJ
telegraph noise) can replace the ideal generator without touching weights.
"""

import torch
from torch import nn


class _StochasticSignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i, u):
        ctx.save_for_backward(i)
        p = 0.5 * (1.0 + torch.tanh(i))
        return torch.where(u < p, 1.0, -1.0)

    @staticmethod
    def backward(ctx, grad_out):
        (i,) = ctx.saved_tensors
        # d E[s] / dI = d tanh(I)/dI = sech^2(I)
        return grad_out * (1.0 - torch.tanh(i) ** 2), None


def stochastic_sign(i: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return _StochasticSignSTE.apply(i, u)


class SBNN(nn.Module):
    """MLP with stochastic bipolar hidden activations and a real-valued readout."""

    def __init__(self, dims=(784, 256, 128, 10)):
        super().__init__()
        self.dims = tuple(dims)
        self.layers = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])
        )

    @property
    def hidden_sizes(self):
        return self.dims[1:-1]

    def forward(self, x: torch.Tensor, source=None) -> torch.Tensor:
        """One stochastic pass. `source` supplies the randomness at inference;
        None means ideal torch randomness (used during training)."""
        h = x
        for li, layer in enumerate(self.layers[:-1]):
            i = layer(h)
            if source is None:
                u = torch.rand_like(i)
                h = stochastic_sign(i, u)
            else:
                p = 0.5 * (1.0 + torch.tanh(i))
                h = source.sample(p, layer=li)
        return self.layers[-1](h)
