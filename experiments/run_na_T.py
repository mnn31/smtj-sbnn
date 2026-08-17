"""E4b: correlation-aware models across the T grid (streaming) — the
throughput frontier lives at small T, and E4 only measured T=32.
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
OUT = ROOT / "results" / "noise_aware_T"

TRAIN_R0 = [0.5, 0.2, 0.1]
EVAL_R0 = [2, 1, 0.5, 0.2, 0.1, 0.05]
T_GRID = [1, 2, 4, 8, 16]
NOISE_SEEDS = [0, 1, 2]


def main(dataset="mnist"):
    OUT.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(load_flat(dataset, train=False), batch_size=1024)
    for tr0 in TRAIN_R0:
        model = load_checkpoint(ROOT / "results" / f"sbnn_{dataset}_na_r{tr0}.pt")
        for er0 in EVAL_R0:
            for T in T_GRID:
                for ns in NOISE_SEEDS:
                    path = OUT / (f"{dataset}_train{tr0}_eval{er0}"
                                  f"_T{T}_ns{ns}_stream.json")
                    if path.exists():
                        continue
                    src = TelegraphSource(r0=er0, noise_seed=ns, streaming=True)
                    acc = evaluate(model, loader, src, T=T)
                    path.write_text(json.dumps(
                        {"dataset": dataset, "train_r0": tr0, "eval_r0": er0,
                         "T": T, "noise_seed": ns, "acc": acc,
                         "streaming": True}))
                    print(f"train{tr0} eval{er0} T={T} ns={ns}: {acc:.4f}",
                          flush=True)


if __name__ == "__main__":
    main()
