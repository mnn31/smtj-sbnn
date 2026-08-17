import numpy as np
import pytest
import torch

from smtjnn.model import SBNN, stochastic_sign
from smtjnn.rng import IdealSource, LFSRSource


def test_stochastic_sign_matches_probability():
    torch.manual_seed(0)
    i = torch.full((200_000,), 0.5)
    u = torch.rand_like(i)
    s = stochastic_sign(i, u)
    assert set(s.unique().tolist()) <= {-1.0, 1.0}
    p_emp = (s == 1).float().mean().item()
    p_true = 0.5 * (1 + np.tanh(0.5))
    assert abs(p_emp - p_true) < 0.005


def test_ste_gradient_is_sech2():
    i = torch.tensor([0.3, -1.2], requires_grad=True)
    u = torch.tensor([0.0, 0.99])  # deterministic outcomes
    s = stochastic_sign(i, u)
    s.sum().backward()
    expected = 1 - np.tanh([0.3, -1.2]) ** 2
    assert np.allclose(i.grad.numpy(), expected, atol=1e-6)


def test_model_forward_shapes_and_sources():
    model = SBNN((20, 16, 8, 4))
    x = torch.randn(5, 20)
    out = model(x)
    assert out.shape == (5, 4)
    out2 = model(x, source=IdealSource(seed=1))
    assert out2.shape == (5, 4)


def test_ideal_source_uniformity():
    src = IdealSource(seed=3)
    p = torch.full((1000, 100), 0.25)
    s = src.sample(p, layer=0)
    frac = (s == 1).float().mean().item()
    assert abs(frac - 0.25) < 0.01


@pytest.mark.parametrize("bits", [2, 4, 8, 16])
def test_lfsr_period_and_range(bits):
    src = LFSRSource(bits=bits, seed=0)
    p = torch.full((1, 1), 0.5)
    seen = []
    for _ in range(2 ** bits + 5):
        src.sample(p, layer=0)
        seen.append(int(src.state[0][0, 0]))
    # maximal-length: period 2^bits - 1, state never 0
    assert 0 not in seen
    assert len(set(seen[: 2 ** bits - 1])) == 2 ** bits - 1
    assert seen[0] == seen[2 ** bits - 1]


def test_lfsr_bias_at_half():
    # u < 0.5 with u uniform on the 2^k-1 nonzero states: near-half firing
    src = LFSRSource(bits=8, seed=1)
    p = torch.full((64, 64), 0.5)
    fracs = [(src.sample(p, layer=0) == 1).float().mean().item()
             for _ in range(255)]
    assert abs(np.mean(fracs) - 0.5) < 0.01


def test_state_persists_across_calls():
    src = LFSRSource(bits=8, seed=2)
    p = torch.full((2, 3), 0.5)
    src.sample(p, layer=0)
    s1 = src.state[0].copy()
    src.sample(p, layer=0)
    assert not np.array_equal(s1, src.state[0])
    src.reset_batch()
    assert src.state == {}


def test_noise_aware_training_gradient_flows():
    import torch as _t
    from smtjnn.rng import IdealSource
    model = SBNN((10, 8, 4))
    model.train()
    x = _t.randn(3, 10)
    out = model(x, source=IdealSource(seed=0))
    out.sum().backward()
    grads = [l.weight.grad for l in model.layers]
    assert all(g is not None and g.abs().sum() > 0 for g in grads)


def test_eval_mode_returns_pure_samples():
    # in eval mode the hidden activations are the raw ±1 device samples,
    # with no straight-through surrogate attached
    from smtjnn.rng import IdealSource
    model = SBNN((10, 8, 4))
    model.eval()
    x = torch.randn(3, 10)
    captured = {}
    orig = IdealSource.sample

    def spy(self, p, layer):
        s = orig(self, p, layer)
        captured[layer] = s
        return s

    IdealSource.sample = spy
    try:
        with torch.no_grad():
            out = model(x, source=IdealSource(seed=0))
    finally:
        IdealSource.sample = orig
    assert not out.requires_grad
    assert set(captured[0].unique().tolist()) <= {-1.0, 1.0}
