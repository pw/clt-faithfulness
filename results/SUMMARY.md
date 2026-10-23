# Do cross-layer transcoders skip the model's real computation? A causal test

DRAFT — fixed-penalty sweep still running (section C).

## Question
Lange et al. (ICML 2026, "Cross-Layer Transcoders are incentivized to learn Unfaithful Circuits")
argue that a CLT's sparsity loss rewards rewriting a multi-step computation A→B→C as direct
one-hop writes A→C. Their evidence: a toy XOR model, plus per-layer L0 asymmetry and attribution
depth in Gemma-2-2B. Both they and Anthropic's methods paper note there is no ground truth for
real models. Their proposed fixes are untested.

## Instrument (ground truth from the real model)
For every source layer s, resample-ablate MLP_s's output (replace it with its value on a
different held-out text). Measure how each later MLP output y_t responds:
- **real**: the model itself;
- **replacement**: every MLP replaced by its transcoder reconstruction plus the clean-run
  error, held frozen (circuit-tracer's local replacement model).

The replacement is written in delta form, so it is bit-exact with the real model on the clean
run (asserted KL < 1e-4). Any disagreement under intervention therefore comes from the
transcoder's features alone.

Metrics per (s, t), averaged over 64 held-out sequences × 128 tokens (position 0 excluded):
- **cos**: mean token-level cosine between replacement Δy_t and real Δy_t. Is the direction right?
- **ratio**: ‖Δy_rep‖ / ‖Δy_real‖. Is the magnitude right?

Controls, run every time: POS (real MLPs as the "replacement") must give cos = ratio = 1. NEG
(MLPs frozen at clean) must give 0. LIN (each MLP replaced by a ridge-regression linear map) is
the floor any transcoder should beat.

A CLT-charitable variant ("write+source") also swaps the CLT's layer-s features to their
donor values, so the CLT's direct writes into later layers change too. That gives the CLT credit
for routing the dependence through cross-layer decoders.

## A. Public transcoders (what circuit-tracer users actually get)
Adjacent layers (t = s + 1):

| model | transcoder | FVU (mean) | cos | ratio | write energy kept in own layer |
|---|---|---|---|---|---|
| Llama-3.2-1B | CLT mntss/clt-llama-3.2-1b-524k | 0.26 | **0.29** (write+source 0.34) | 0.29 | 0.32 |
| | PLT mntss/transcoder-Llama-3.2-1B | 0.15 | 0.75 | 0.88 | 1 |
| | PLT skip path only (no features) | 0.61 | 0.62 | 0.72 | 1 |
| Gemma-2-2B | CLT mntss/clt-gemma-2-2b-426k | 0.28 | **0.32** | 0.38 | 0.24 |
| | PLT GemmaScope 16k (no skip) | 0.41 | 0.43 | 0.59 | 1 |

- The CLT gets the direction of the adjacent-layer response wrong far more often than the PLT, in
  both models. That still holds when the CLT also gets credit for its cross-layer writes.
- Reconstruction quality does not explain it. On Gemma the CLT reconstructs better than the PLT
  and is still less faithful. On Llama a bare linear skip map, FVU 0.61, beats the CLT.
- The public CLTs keep only 24–32% of their write energy in the source layer. The rest goes to
  later layers. Their per-layer L0 is strongly front-loaded: Llama 29→1, Gemma 37→2.
- Secondary: the Gemma CLT replacement diverges under early-layer interventions. Its late-layer
  responses run 10²–10⁴× the real ones.

## B. Controlled sweep, GPT-2 small (our transcoders, matched total L0, 300M tokens, 8,192 feats/layer)
Adjacent layers:

| transcoder | L0/token | cos | ratio | own-layer write |
|---|---|---|---|---|
| LIN floor | – | 0.32 | 0.43 | – |
| PLT (2 seeds) | 198 / 187 | 0.40 / 0.40 | 0.91 / 0.91 | 1 |
| PLT sparse | 117 | 0.30 | 0.90 | 1 |
| CLT (2 seeds) | 192 / 171 | 0.40 / 0.40 | 0.69 / 0.69 | 0.73 / 0.71 |
| CLT sparse | 94 | 0.31 | 0.65 | 0.64 |
| CLT dense | 436 | 0.50 | 0.73 | 0.78 |
| CLT sum-of-norms (fix 1) | 156 | 0.39 | 0.68 | 0.85 |
| CLT window-3 (fix 2) | 207 | 0.40 | 0.68 | – |

These were trained with an L0 controller that drove the sparsity coefficient down to about 0.004,
so sparsity came from the thresholds rather than from the penalty. In that regime the CLTs keep
most of their writing local. They match the PLT on direction and damp the response magnitude by
about 25%. Seeds agree to 0.01.

## C. Fixed-penalty sweep (θ₀ = 0.1, so sparsity has to come from the penalty)
Adjacent layers. L0 = total active features per token.

| transcoder | λ | L0 | cos | ratio | own-layer write |
|---|---|---|---|---|---|
| CLT concat | 0.1 | 29 | 0.13 | 0.54 | 0.32 |
| CLT concat | 0.3 | 7 | 0.16 | 0.32 | 0.23 |
| CLT concat | 1.0 | 3 | 0.17 | 0.09 | 0.18 |
| CLT sum-of-norms | 0.3 | 6 | 0.14 | 0.16 | 0.26 |
| CLT sum-of-norms | 1.0 | 2 | 0.11 | 0.03 | 0.23 |
| PLT | 0.3 | 15 | 0.18 | 0.52 | 1 |

- Penalty-driven sparsity reproduces the public CLTs' regime: front-loaded L0 and 18–32%
  own-layer writing. It also reproduces their adjacent-layer direction deficit.
- At these extreme sparsities, a penalty-driven PLT is nearly as unfaithful (0.18), so this grid
  cannot separate cross-layer routing from sparsity. Moderate-sparsity runs are pending:
  CLT λ 0.01 / 0.003 (+seed), matched PLT λ 0.01 / 0.003, sum-of-norms λ 0.03 / 0.01 / 0.003.
- Sum-of-norms (fix 1) does not raise own-layer writing at matched penalty. It is just a
  stronger penalty, about 10× sparser at the same λ.
