Dear Editors of IEEE Access,

I am submitting "Overclocking the p-bit: Stochastic Binary Neural Network
Inference with Autocorrelated sMTJ Randomness" for consideration as a
research article.

Stochastic magnetic tunnel junctions produce physically random bits at
roughly five to six orders of magnitude lower energy than software
pseudorandom generation, making them leading entropy sources for
probabilistic-bit neural hardware. A standard design rule requires
sampling these devices no faster than twice their mean dwell time;
existing systems obey it by construction, and its application-level cost
has not been measured. This paper measures that cost: a continuous sweep
of the sampling-to-correlation-time ratio, mapped to classification
accuracy across operating modes, device-variability models, and training
regimes. The main findings: batch-style inference tolerates sampling
200x past the rule with under one accuracy point of loss (given a
measured ~4-correlation-time per-input settle); steady-state streaming
collapses to chance through cross-input state leakage; device dispersion
does not rescue it, but correlation-aware training recovers most of the
loss, extending the accuracy-throughput frontier 2-8x past the
rule-compliant optimum.

The study is fully computational, with results replicated across five
independently trained networks and two datasets; the simulation code and
all per-run records will be made public on publication. Generative AI
assistance was used in this work and is disclosed in the Acknowledgment
section per IEEE policy. The manuscript is original and not under
consideration elsewhere.

Sincerely,
Manan Gupta
The Harker School, San Jose, CA
mnn@yogins.com
