import numpy as np
import torch

from smtjnn.smtj import TelegraphSource


def _run_chain(src, p_scalar, n_dev, n_steps):
    p = torch.full((1, n_dev), p_scalar, dtype=torch.float64)
    out = np.empty((n_steps, n_dev))
    for t in range(n_steps):
        out[t] = src.sample(p, layer=0).numpy()[0]
    return out


def test_stationary_marginal_matches_pbit_law():
    # long-run fraction of +1 must equal p_eq regardless of correlation
    for r0 in [np.inf, 1.0, 0.2]:
        src = TelegraphSource(r0=r0, noise_seed=1)
        traj = _run_chain(src, 0.73, n_dev=400, n_steps=4000)
        frac = (traj > 0).mean()
        assert abs(frac - 0.73) < 0.01, (r0, frac)


def test_autocorrelation_decays_at_theoretical_rate():
    # at p=0.5 (I=0, cosh=1): ACF(k) = exp(-k * r0)
    r0 = 0.5
    src = TelegraphSource(r0=r0, noise_seed=2)
    traj = _run_chain(src, 0.5, n_dev=2000, n_steps=3000)
    x = traj - traj.mean()
    for k in [1, 2, 4]:
        acf = (x[:-k] * x[k:]).mean() / x.var()
        assert abs(acf - np.exp(-k * r0)) < 0.02, (k, acf)


def test_ideal_limit_is_iid():
    src = TelegraphSource(r0=np.inf, noise_seed=3)
    traj = _run_chain(src, 0.5, n_dev=2000, n_steps=500)
    x = traj - traj.mean()
    acf1 = (x[:-1] * x[1:]).mean() / x.var()
    assert abs(acf1) < 0.01


def test_frozen_limit_barely_flips():
    src = TelegraphSource(r0=0.001, noise_seed=4)
    traj = _run_chain(src, 0.5, n_dev=500, n_steps=100)
    flips = (traj[1:] != traj[:-1]).mean()
    assert flips < 0.01


def test_saturated_neurons_decorrelate_faster():
    # cosh(I) speeds decorrelation: ACF at p=0.95 decays faster than at p=0.5
    r0 = 0.3

    def acf1(p_scalar, seed):
        src = TelegraphSource(r0=r0, noise_seed=seed)
        traj = _run_chain(src, p_scalar, n_dev=3000, n_steps=2000)
        x = traj - traj.mean()
        return (x[:-1] * x[1:]).mean() / x.var()

    assert acf1(0.95, 5) < acf1(0.5, 6) - 0.05


def test_chip_seed_fixes_device_parameters():
    a = TelegraphSource(r0=0.5, sigma_delta=1.0, chip_seed=7, noise_seed=1)
    b = TelegraphSource(r0=0.5, sigma_delta=1.0, chip_seed=7, noise_seed=99)
    c = TelegraphSource(r0=0.5, sigma_delta=1.0, chip_seed=8, noise_seed=1)
    p = torch.full((2, 50), 0.5, dtype=torch.float64)
    for src in (a, b, c):
        src.sample(p, layer=0)
        src.sample(p, layer=0)
    assert np.allclose(a.rate_factor[0], b.rate_factor[0])
    assert not np.allclose(a.rate_factor[0], c.rate_factor[0])


def test_streaming_vs_reset_state_behavior():
    p = torch.full((2, 10), 0.5, dtype=torch.float64)
    s1 = TelegraphSource(r0=0.5, noise_seed=1, streaming=False)
    s1.sample(p, layer=0)
    s1.reset_batch()
    assert s1.state == {}
    s2 = TelegraphSource(r0=0.5, noise_seed=1, streaming=True)
    s2.sample(p, layer=0)
    s2.reset_batch()
    assert 0 in s2.state
