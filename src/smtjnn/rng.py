"""Pluggable inference-time randomness sources.

Contract: `sample(p, layer)` receives firing probabilities p in [0,1] with
shape [B, N] and returns bipolar samples (+1/-1) of the same shape. Sources
are stateful: each (layer, neuron, batch-lane) owns an independent physical
generator, and state persists across repeated calls (T inference passes),
mirroring hardware where the same devices are queried every pass.
"""

import numpy as np
import torch


class IdealSource:
    """i.i.d. uniform randomness from a seeded PCG64 — the software baseline."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset_batch(self):
        pass

    def sample(self, p: torch.Tensor, layer: int) -> torch.Tensor:
        u = torch.from_numpy(
            self.rng.random(tuple(p.shape), dtype=np.float64)
        ).to(p.dtype)
        return torch.where(u < p, 1.0, -1.0)


# Maximal-length Fibonacci LFSR tap positions (1-indexed), standard tables.
_TAPS = {2: (2, 1), 3: (3, 2), 4: (4, 3), 6: (6, 5), 8: (8, 6, 5, 4),
         10: (10, 7), 12: (12, 11, 10, 4), 16: (16, 14, 13, 11)}


class LFSRSource:
    """One k-bit maximal-length LFSR per (neuron, batch-lane).

    The full k-bit state is used as the uniform variate u = state / 2^k,
    so short LFSRs give both coarse quantization and a short period —
    the digital-degradation baseline.
    """

    def __init__(self, bits: int, seed: int = 0):
        if bits not in _TAPS:
            raise ValueError(f"no tap table for {bits}-bit LFSR")
        self.bits = bits
        self.seed = seed
        self.taps = _TAPS[bits]
        self.state: dict[int, np.ndarray] = {}

    def reset_batch(self):
        self.state.clear()

    def _init_state(self, shape, layer):
        rng = np.random.default_rng(self.seed + 7919 * layer)
        # nonzero initial states
        return rng.integers(1, 2 ** self.bits, size=shape, dtype=np.uint32)

    def sample(self, p: torch.Tensor, layer: int) -> torch.Tensor:
        shape = tuple(p.shape)
        s = self.state.get(layer)
        if s is None or s.shape != shape:
            s = self._init_state(shape, layer)
        fb = np.zeros(shape, dtype=np.uint32)
        for t in self.taps:
            fb ^= (s >> (t - 1)) & 1
        s = ((s << 1) | fb) & ((1 << self.bits) - 1)
        # LFSR state never hits 0, so remap the 1..2^k-1 range onto [0,1)
        self.state[layer] = s
        u = torch.from_numpy((s - 1).astype(np.float64) / (2 ** self.bits - 1))
        return torch.where(u.to(p.dtype) < p, 1.0, -1.0)
