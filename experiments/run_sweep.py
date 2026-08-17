"""E1/E2: accuracy vs sampling ratio r0, T-averaging, reset vs streaming.

Writes one JSON per run to results/sweeps/. Resumable: existing configs are
skipped, so the sweep can be re-launched after interruption.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from smtjnn.data import load_flat
from smtjnn.infer import evaluate
from smtjnn.rng import IdealSource
from smtjnn.smtj import TelegraphSource
from smtjnn.train import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "sweeps"

R0_GRID = [np.inf, 10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02]
T_GRID = [1, 8, 32]
NOISE_SEEDS = [0, 1, 2]


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def tag(cfg):
    r0 = "inf" if np.isinf(cfg["r0"]) else cfg["r0"]
    return (f"{cfg['dataset']}_r{r0}_T{cfg['T']}_"
            f"{'stream' if cfg['streaming'] else 'reset'}_"
            f"sd{cfg['sigma_delta']}_chip{cfg['chip_seed']}_ns{cfg['noise_seed']}"
            + cfg.get("extra", ""))


def run_one(model, loader, cfg, sha):
    path = OUT / (tag(cfg) + ".json")
    if path.exists():
        return None
    src = TelegraphSource(r0=cfg["r0"], sigma_delta=cfg["sigma_delta"],
                          chip_seed=cfg["chip_seed"],
                          noise_seed=cfg["noise_seed"],
                          streaming=cfg["streaming"])
    t0 = time.time()
    acc = evaluate(model, loader, src, T=cfg["T"])
    rec = dict(cfg, acc=acc, seconds=round(time.time() - t0, 1), git=sha)
    rec["r0"] = "inf" if np.isinf(cfg["r0"]) else cfg["r0"]
    path.write_text(json.dumps(rec))
    return rec


def main(dataset="mnist", ckpt="sbnn_mnist_seed0.pt", batch_size=1024):
    OUT.mkdir(parents=True, exist_ok=True)
    sha = git_sha()
    model = load_checkpoint(ROOT / "results" / ckpt)
    loader = DataLoader(load_flat(dataset, train=False), batch_size=batch_size)

    # E1: ideal-source reference across T (IdealSource, sanity anchor)
    for T in [1, 2, 4, 8, 16, 32, 64]:
        p = OUT / f"{dataset}_ideal_T{T}.json"
        if not p.exists():
            acc = evaluate(model, loader, IdealSource(seed=0), T=T)
            p.write_text(json.dumps({"dataset": dataset, "source": "ideal",
                                     "T": T, "acc": acc, "git": sha}))
            print(f"ideal T={T}: {acc:.4f}", flush=True)

    # E2: correlation sweep
    n = done = 0
    for r0 in R0_GRID:
        for T in T_GRID:
            for streaming in [False, True]:
                for ns in NOISE_SEEDS:
                    cfg = dict(dataset=dataset, r0=r0, T=T, streaming=streaming,
                               sigma_delta=0.0, chip_seed=0, noise_seed=ns)
                    rec = run_one(model, loader, cfg, sha)
                    n += 1
                    if rec:
                        done += 1
                        print(f"[{n}] {tag(cfg)} acc={rec['acc']:.4f} "
                              f"({rec['seconds']}s)", flush=True)
    print(f"sweep complete: {done} new / {n} total", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--ckpt", default="sbnn_mnist_seed0.pt")
    args = ap.parse_args()
    main(dataset=args.dataset, ckpt=args.ckpt)
