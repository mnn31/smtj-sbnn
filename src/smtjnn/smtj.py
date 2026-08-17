"""Telegraph-noise sMTJ neuron model.

Each neuron is a two-state fluctuator with Néel–Arrhenius escape rates
biased by its input so the stationary law is the p-bit activation
P(+1) = (1 + tanh I)/2, while the *time structure* is controlled by the
sampling ratio r0 = Δt · 2 f0 (Δt = query interval, f0 = nominal attempt
rate at zero bias). r0 ≫ 1 recovers i.i.d. draws; r0 ≪ 1 returns stale,
strongly autocorrelated samples.

Exact discrete-time propagator of the two-state Markov process:
    P(s' = +1 | s) = p_eq + (1[s=+1] − p_eq) · exp(−Δt/τ_corr)
with Δt/τ_corr = r0 · exp(−δΔ) · cosh(I), where δΔ ~ N(0, σ_Δ) is the
per-device energy-barrier offset in kT units (fixed per chip).
"""

import numpy as np
import torch


class TelegraphSource:
    """Stateful sMTJ randomness source implementing the rng.py contract.

    Parameters
    ----------
    r0 : sampling ratio Δt·2f0 at zero bias for a nominal device.
         np.inf → i.i.d. (ideal) limit.
    sigma_delta : std of per-device barrier offset δΔ in kT (0 = identical devices).
    chip_seed : seeds the fixed device parameters (one 'chip').
    noise_seed : seeds the thermal flip randomness.
    streaming : if True, reset_batch() keeps device state (continuous operation
         across images); if False, state re-equilibrates per batch.
    """

    def __init__(self, r0: float, sigma_delta: float = 0.0, chip_seed: int = 0,
                 noise_seed: int = 0, streaming: bool = False):
        if r0 <= 0:
            raise ValueError("r0 must be positive (use np.inf for ideal)")
        self.r0 = float(r0)
        self.sigma_delta = float(sigma_delta)
        self.chip_seed = chip_seed
        self.streaming = streaming
        self.rng = np.random.default_rng(noise_seed)
        self.state: dict[int, np.ndarray] = {}   # layer -> ±1 float64 [B, N]
        self.rate_factor: dict[int, np.ndarray] = {}  # layer -> e^{-δΔ} [N]

    def reset_batch(self):
        if not self.streaming:
            self.state.clear()

    def _rate_factor(self, n: int, layer: int) -> np.ndarray:
        rf = self.rate_factor.get(layer)
        if rf is None or rf.shape[0] != n:
            crng = np.random.default_rng(self.chip_seed + 104729 * layer)
            delta = crng.normal(0.0, self.sigma_delta, size=n)
            rf = np.exp(-delta)
            self.rate_factor[layer] = rf
        return rf

    def sample(self, p: torch.Tensor, layer: int) -> torch.Tensor:
        p_np = p.detach().cpu().numpy().astype(np.float64)
        b, n = p_np.shape
        s = self.state.get(layer)
        if s is None or s.shape != (b, n):
            # first query: device has been idle long enough to equilibrate
            s = np.where(self.rng.random((b, n)) < p_np, 1.0, -1.0)
        else:
            if np.isinf(self.r0):
                lam = 0.0
            else:
                i_pre = np.arctanh(np.clip(2.0 * p_np - 1.0, -1 + 1e-12, 1 - 1e-12))
                dt_over_tau = (self.r0 * self._rate_factor(n, layer)[None, :]
                               * np.cosh(i_pre))
                lam = np.exp(-dt_over_tau)
            p_next = p_np + ((s > 0).astype(np.float64) - p_np) * lam
            s = np.where(self.rng.random((b, n)) < p_next, 1.0, -1.0)
        self.state[layer] = s
        return torch.from_numpy(s).to(dtype=p.dtype, device=p.device)
