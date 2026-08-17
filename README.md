# smtj-sbnn

Simulation study: **stochastic binary neural network (SBNN) inference under
autocorrelated sMTJ randomness** — what happens to classification accuracy
when probabilistic-bit devices are sampled faster than their intrinsic
correlation time, and what recovers the loss.

Paper: `paper/main.tex` (IEEE format). Design rationale: `docs/design.md`.
PDF-verified physics parameters and citations: `docs/physics_params.md`.

## Layout

```
src/smtjnn/        model (SBNN + STE), telegraph sMTJ source, protocols, training
tests/             physics + protocol validation (pytest)
experiments/       run_v2.py (all phases), analysis / figure scripts
results/v2/        one JSON per (config, seed), each stamped with its git SHA
results/*.pt       trained checkpoints (tracked)
paper/             LaTeX source, refs.bib, figs/
```

## Reproduce

```bash
pip install -e ".[dev]"
python -m pytest tests/          # physics & protocol validation
python experiments/run_v2.py --phase 0   # train all models (val split; test untouched)
python experiments/run_v2.py --phase 1   # ideal-randomness anchors
python experiments/run_v2.py --phase 2   # E2: sampling-ratio sweep (settle + streaming + const-tau)
python experiments/run_v2.py --phase 3   # E3: device-variability interaction (both centerings)
python experiments/run_v2.py --phase 4   # E4: correlation-aware models (val for selection, test for reporting)
python experiments/run_v2.py --phase 5   # steady-state / lane-count invariance check
python experiments/run_v2.py --phase 2 --dataset fashion   # replication
```

Runs are resumable; a result is reused only if its recorded git SHA matches
HEAD, so data can never silently mix code versions. Everything is CPU-only;
the full suite takes a few hours on a laptop.

## Key modeling points

- The sMTJ neuron is a two-state telegraph process with input-biased
  Néel–Arrhenius rates; stationary law = p-bit activation `(1+tanh I)/2`;
  exact discrete-time propagator (piecewise-constant input per interval).
- Sampling ratio `r0 = Δt·2f0` (query interval over nominal zero-bias
  correlation time). The Daniels et al. 2τ rule is stated in **mean dwell
  time** = 2·τ_corr at zero bias, so rule-compliant operation is `r0 ≥ 4`.
- **Settle mode** (batch-style): each input presentation begins with a
  finite relaxation interval (`settle_ratio·τ`) from the device's prior
  state under the newly applied input — idle devices equilibrate to p=1/2,
  not to the upcoming input's law. Throughput must charge the settle.
- **Streaming mode**: state persists across inputs; evaluation uses true
  per-lane sequential streams with unscored warm-up images so scored
  accuracy is steady-state (lane-count invariance verified in phase 5).
- Device-to-device barrier spread supports `median` and `mean-tau`
  centerings; the latter isolates dispersion from the Jensen mean-slowdown.
- `bias_independent_tau=True` gives the easy-plane-like variant (no
  cosh(I) rate speedup) to separate device-class-specific effects.

## AI disclosure

Simulation code, experiment orchestration, and manuscript drafting were done
with substantial assistance from AI tools (Anthropic Claude), under the
author's direction and review; see the paper's Acknowledgments.
