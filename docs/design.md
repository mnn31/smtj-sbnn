# Design: Overclocking the p-bit — SBNN inference under autocorrelated sMTJ randomness

## Research question

Every sMTJ p-bit paper treats the decorrelation rule (sample interval > 2× device
correlation time, Camsari et al. arXiv 2304.05949) as a hard design constraint.
We treat it as a **continuous knob**: sweep the sampling ratio
`r = Δt_sample / τ_corr` from the i.i.d. regime (r ≫ 1) down to heavy staleness
(r ≪ 1) and measure where stochastic-binary-NN classification actually fails.
Deliverable: accuracy-vs-r curves and a throughput–accuracy Pareto that tells a
hardware designer how far past the 2τ rule they can clock a given device.

Novelty basis (adversarial scan, 2026-08-17): the temporal axis is unclaimed;
static D2D variation → accuracy is crowded (Zand/DeMara 1811.11390, Wood
2002.00897, Micromachines 16:133); PRNG-quality → MNIST is taken (2510.25269).

## Physics model: the sMTJ *is* the neuron

Two-state telegraph process s ∈ {−1,+1} with Néel–Arrhenius escape rates
modulated by the input I (dimensionless, in units where p-bit statistics hold):

    k(−→+) = f · e^{+I},   k(+→−) = f · e^{−I},   f = f0 · e^{−δΔ}

- Stationary distribution: P(+1) = e^I / (e^I + e^{−I}) = (1 + tanh I)/2 — exactly
  the p-bit activation. The mean behavior is untouched; only *time structure* changes.
- Correlation time: τ_corr(I) = 1/(k₊₋ + k₋₊) = 1/(2 f cosh I). Saturated neurons
  decorrelate faster — an emergent, physical effect we keep.
- Exact discrete-time propagator (no timestep error):
      P(s_{t+Δt} = +1 | s_t) = p_eq + (𝟙[s_t = +1] − p_eq) · e^{−Δt/τ_corr}
- Control knob: **r0 = Δt · 2 f0** = sampling ratio at I = 0 for a nominal device.
  Then Δt/τ_corr = r0 · e^{−δΔ} · cosh I.
- Device variation: barrier offsets δΔ_i ~ N(0, σ_Δ) in kT units, fixed per
  (layer, neuron) for a given chip seed → log-normal spread of correlation times.
  Grounded in the ~1 kT target barrier and fabrication spread reported for real
  arrays (Micromachines 16(2):133 2025; Borders 2019 devices span 29 μs vs 27 ms).

State persists across the T inference passes of one image and (in streaming mode)
across consecutive images — modeling real hardware where you cannot pause the
device between queries.

## Experiment matrix (MNIST primary, Fashion-MNIST generality check)

- **E1 baseline**: ideal i.i.d. randomness, accuracy vs T ∈ {1,2,4,8,16,32,64}; 5 training seeds.
- **E2 correlation sweep (headline)**: accuracy vs r0 ∈ {∞, 10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02}
  × T ∈ {1, 8, 32}, reset-per-image vs streaming (no reset). Hypotheses:
  T=1 with equilibrium init is exactly Bernoulli(p_eq) → flat; T>1 averaging loses
  effective samples as r0 drops (T_eff ≈ ...); streaming adds cross-image leakage.
- **E3 heterogeneity interaction**: fixed r0 ∈ {0.2, 0.5, 1}, sweep σ_Δ ∈ {0, 0.25, 0.5, 1, 1.5} kT,
  ≥5 chip seeds. Tests the desynchronization hypothesis (variation may *help*).
- **E4 correlation-aware training**: retrain with the telegraph source active at
  r0 ∈ {0.1, 0.2, 0.5}; how much of the E2 penalty is recovered.
- **E5 throughput Pareto**: map r0 → images/s for real devices (Safranski 2021
  ns-scale vs Borders 2019 27 ms) and plot accuracy vs throughput; headline figure.

Rigor floor: full 10k test set, ≥5 seeds per point, mean ± std, all sweep configs
logged to JSON with git SHA; every physics parameter cited to a primary PDF we
have actually opened (two hallucinated web-summary claims were caught during the
novelty scan — nothing enters the paper without PDF verification).

## Dropped axes (deliberate)

- Temperature/bias drift: characterized elsewhere (Rehm 2024 dp/dT = 0.0016/K),
  accuracy propagation is incremental → future-work paragraph only.
- Static sigmoid slope/offset variation: crowded (1811.11390, Micromachines) → cite, don't redo.
- LFSR/digital defects: kept only as a baseline curve linking to prior work.

## Venue targets

IEEE Access or IJHSR (primary), arXiv cs.ET preprint with mentor endorsement.
