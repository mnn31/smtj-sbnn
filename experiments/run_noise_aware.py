"""E4: correlation-aware training — retrain with the telegraph source active,
then evaluate across the r0 grid. How much of the E2 penalty is recovered?
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from smtjnn.data import load_flat
from smtjnn.infer import evaluate
from smtjnn.smtj import TelegraphSource
from smtjnn.train import train, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "noise_aware"

TRAIN_R0 = [0.5, 0.2, 0.1]
EVAL_R0 = [np.inf, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02]
T = 32
NOISE_SEEDS = [0, 1, 2]


def main(dataset="mnist"):
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    loader = DataLoader(load_flat(dataset, train=False), batch_size=1024)
    for tr0 in TRAIN_R0:
        name = f"sbnn_{dataset}_na_r{tr0}.pt"
        ckpt = ROOT / "results" / name
        if not ckpt.exists():
            # training runs on CPU: the telegraph source is numpy-side, and
            # constant device transfers erase any MPS gain
            src = TelegraphSource(r0=tr0, noise_seed=123, streaming=True)
            train(dataset=dataset, epochs=30, seed=0, device="cpu",
                  ckpt_name=name, source=src)
        model = load_checkpoint(ckpt)
        for streaming in [True, False]:
            mode = "stream" if streaming else "reset"
            for er0 in EVAL_R0:
                for ns in NOISE_SEEDS:
                    er0_tag = "inf" if np.isinf(er0) else er0
                    path = OUT / (f"{dataset}_train{tr0}_eval{er0_tag}"
                                  f"_ns{ns}_{mode}.json")
                    if path.exists():
                        continue
                    esrc = TelegraphSource(r0=er0, noise_seed=ns,
                                           streaming=streaming)
                    acc = evaluate(model, loader, esrc, T=T)
                    path.write_text(json.dumps(
                        {"dataset": dataset, "train_r0": tr0,
                         "eval_r0": str(er0_tag), "noise_seed": ns, "T": T,
                         "acc": acc, "streaming": streaming}))
                    print(f"[{mode}] train_r0={tr0} eval_r0={er0_tag} "
                          f"ns={ns}: {acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
