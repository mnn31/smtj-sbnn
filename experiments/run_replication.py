"""Replications, run sequentially:
1. Fashion-MNIST: full E1+E2 grid (generality across datasets).
2. MNIST training seeds 1-4: reduced key grid (findings not a seed-0 artifact).
3. MNIST seed 0: extended T grid (Pareto needs fine T resolution).
"""

import numpy as np
from torch.utils.data import DataLoader

from run_sweep import OUT, git_sha, main as sweep_main, run_one
from smtjnn.data import load_flat
from smtjnn.train import load_checkpoint
from run_sweep import ROOT


def phase2_mnist_seeds():
    sha = git_sha()
    loader = DataLoader(load_flat("mnist", train=False), batch_size=1024)
    for seed in [1, 2, 3, 4]:
        model = load_checkpoint(ROOT / "results" / f"sbnn_mnist_seed{seed}.pt")
        for r0 in [np.inf, 1, 0.2, 0.05]:
            for T in [1, 32]:
                for streaming in [False, True]:
                    cfg = dict(dataset="mnist", r0=r0, T=T, streaming=streaming,
                               sigma_delta=0.0, chip_seed=0, noise_seed=0,
                               extra=f"_ts{seed}")
                    rec = run_one(model, loader, cfg, sha)
                    if rec:
                        print(f"seed{seed} r0={rec['r0']} T={T} "
                              f"{'stream' if streaming else 'reset'}: "
                              f"{rec['acc']:.4f}", flush=True)


def phase3_extended_T():
    sha = git_sha()
    loader = DataLoader(load_flat("mnist", train=False), batch_size=1024)
    model = load_checkpoint(ROOT / "results" / "sbnn_mnist_seed0.pt")
    for r0 in [2, 1, 0.5, 0.2, 0.1, 0.05, 0.02]:
        for T in [2, 4, 16, 64]:
            for streaming in [False, True]:
                for ns in [0, 1, 2]:
                    cfg = dict(dataset="mnist", r0=r0, T=T, streaming=streaming,
                               sigma_delta=0.0, chip_seed=0, noise_seed=ns)
                    rec = run_one(model, loader, cfg, sha)
                    if rec:
                        print(f"extT r0={r0} T={T} "
                              f"{'stream' if streaming else 'reset'} ns={ns}: "
                              f"{rec['acc']:.4f}", flush=True)


if __name__ == "__main__":
    print("=== phase 1: fashion full grid ===", flush=True)
    sweep_main(dataset="fashion", ckpt="sbnn_fashion_seed0.pt")
    print("=== phase 2: mnist training seeds ===", flush=True)
    phase2_mnist_seeds()
    print("=== phase 3: extended T ===", flush=True)
    phase3_extended_T()
    print("replication complete", flush=True)
