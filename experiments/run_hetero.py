"""E3: does device-to-device barrier spread help or hurt at fixed r0?

At fixed sampling ratio r0, sweep sigma_delta (kT units) over multiple chip
seeds. Hypothesis: heterogeneity desynchronizes staleness across neurons.
Note sigma_delta preserves each neuron's stationary p (only tau spreads),
so any effect is purely temporal.
"""

import json
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from smtjnn.data import load_flat
from smtjnn.infer import evaluate
from smtjnn.smtj import TelegraphSource
from smtjnn.train import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "hetero"

R0_GRID = [0.2, 0.5, 1.0]
SIGMA_GRID = [0.0, 0.25, 0.5, 1.0, 1.5]
CHIP_SEEDS = [0, 1, 2, 3, 4]
T = 32


def main(dataset="mnist", ckpt="sbnn_mnist_seed0.pt"):
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_checkpoint(ROOT / "results" / ckpt)
    loader = DataLoader(load_flat(dataset, train=False), batch_size=1024)
    for streaming in [True, False]:
        mode = "stream" if streaming else "reset"
        for r0 in R0_GRID:
            for sd in SIGMA_GRID:
                for chip in CHIP_SEEDS:
                    path = OUT / f"{dataset}_r{r0}_sd{sd}_chip{chip}_{mode}.json"
                    if path.exists():
                        continue
                    src = TelegraphSource(r0=r0, sigma_delta=sd, chip_seed=chip,
                                          noise_seed=chip, streaming=streaming)
                    acc = evaluate(model, loader, src, T=T)
                    path.write_text(json.dumps(
                        {"dataset": dataset, "r0": r0, "sigma_delta": sd,
                         "chip_seed": chip, "T": T, "acc": acc,
                         "streaming": streaming}))
                    print(f"[{mode}] r0={r0} sd={sd} chip={chip}: {acc:.4f}",
                          flush=True)


if __name__ == "__main__":
    main()
