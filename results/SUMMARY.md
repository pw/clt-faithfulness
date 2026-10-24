# Do cross-layer transcoders skip the model's real computation? A causal test

GPT-2 small: 28 transcoders. GPT-2 medium replication: 4 transcoders (section G).

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
Adjacent layers. L0 = total active features per token. "own" = write energy kept in the source layer.

| transcoder | λ | L0 | cos | ratio | own |
|---|---|---|---|---|---|
| CLT concat | 0.003 | 704 | 0.60 | 0.77 | 0.82 |
| CLT concat | 0.01 (2 seeds) | 76 / 76 | 0.22 / 0.23 | 0.67 / 0.68 | 0.51 / 0.49 |
| CLT concat | 0.03 | 51 | 0.16 | 0.81 | 0.39 |
| CLT concat | 0.1 | 29 | 0.13 | 0.54 | 0.32 |
| CLT concat | 0.3 | 7 | 0.16 | 0.32 | 0.23 |
| CLT concat | 1.0 | 3 | 0.17 | 0.09 | 0.18 |
| CLT sum-of-norms | 0.003 | 317 | 0.42 | 0.70 | 0.81 |
| CLT sum-of-norms | 0.01 | 140 | 0.19 | 0.69 | 0.60 |
| CLT sum-of-norms | 0.03 | 127 | 0.18 | 0.67 | 0.43 |
| CLT sum-of-norms | 0.3 / 1.0 | 6 / 2 | 0.14 / 0.11 | 0.16 / 0.03 | 0.26 / 0.23 |
| PLT | 0.003 | 1048 | 0.68 | 0.95 | 1 |
| PLT | 0.01 | 117 | 0.30 | 1.00 | 1 |
| PLT | 0.03 | 97 | 0.23 | 0.98 | 1 |
| PLT | 0.3 | 15 | 0.18 | 0.52 | 1 |

## D. The one curve
Fit adjacent-layer cos against ln L0 on all PLTs (L0 ≥ 50; both recipes):
**cos = −0.56 + 0.18 · ln L0**, residuals ≤ 0.03.

Mean residual from that curve:

| family | n | mean residual |
|---|---|---|
| PLT (fit) | 7 | 0.00 |
| controller CLTs | 6 | +0.02 |
| penalty concat CLTs | 4 | 0.00 |
| **penalty sum-of-norms CLTs (fix 1)** | 3 | **−0.11** |

**At matched sparsity, CLTs are as intervention-faithful as per-layer transcoders.** This holds
even though the share of write energy they keep in the source layer ranges from 0.39 to 0.85 and their L0 profiles range
from flat to strongly front-loaded. On this metric, cross-layer routing does not cost
faithfulness. Sparsity does, steeply: about +0.18 cos per e-fold of L0.

The proposed sum-of-norms fix sits on the curve when trained with an L0 target (L0 156, residual
about +0.04) and below it, by about 0.11, when sparsity comes from a fixed penalty. It never helps.

Magnitudes: CLTs under-respond more than PLTs (ratio about 0.7 vs 0.9 at adjacent layers). That
is the one CLT-specific difference that survives.

## E. Public models, re-read with the L0 confound in mind
Layer numbers below follow GemmaScope's (layer_11 is its first sparse layer). Each adjacent-layer
response at target t is paired with the L0 of the features that can respond: CLT source t, PLT layer t.

- **Gemma-2-2B.** GemmaScope's per-layer L0 jumps from ~50–98 (layers 1–10) to ~5–16 (layers
  11–22). The CLT-vs-PLT ordering flips with it: PLT ahead at layers 1–10, CLT ahead at 11–13.
  Regressing per-layer cos on ln L0 plus an architecture indicator, over all 25 target layers:
  **CLT term −0.002 (SE 0.019)**. With layer fixed effects: −0.016 (SE 0.009). The headline gap
  (0.32 vs 0.43) is sparsity. (An earlier draft quoted a hand-picked matched window,
  0.28 vs 0.29. Shifting that window by two layers gives 0.27 vs 0.33, so the regression replaces it.)
- **Llama-3.2-1B. The gap is NOT explained.** The same regression gives a CLT term of
  **−0.42 (SE 0.03)**. The PLT's linear skip path alone, with no features, scores 0.62 averaged
  over all layers (0.65 over the layers with L0 within 2× of the CLT's). The PLT's features add
  only +0.12 over that. So the PLT's lead is mostly its skip connection, but the CLT (0.29) sits
  far below even a featureless linear map, and nothing here accounts for that. Cross-layer writing
  as such doesn't either: it keeps 32% of its write energy in the source layer, while the Gemma CLT
  keeps 24% with no architecture effect. (Our on-curve GPT-2-small CLTs go down to 39%; the one
  GPT-2-medium penalty CLT, at 27%, sat about 0.05 low.) Open hypotheses: its training recipe,
  width, or something about Llama. Reproduce both regressions with `python -m cltf.archreg`
  (pooled −0.42 / layer fixed effects −0.46 for Llama; −0.002 / −0.016 for Gemma).
  Sensitivity: Llama's own fitted sparsity slope is nearly flat (0.03 cos per e-fold of L0).
  Borrowing GPT-2's slope (0.18) instead, sparsity would account for roughly 0.24 of the raw gap,
  leaving about 0.2. Either way a real CLT deficit remains on Llama.

## F. Skip connections (controller recipe, matched L0)
Each transcoder gets a learned linear map x_l → y_l (d×d, zero init), not counted in L0 or the penalty.

| transcoder | L0 | cos | ratio | FVU (mid layers) |
|---|---|---|---|---|
| PLT | 192–198 | 0.40 | 0.91 | ~0.45 |
| PLT + skip | 202 | **0.53** | 0.93 | ~0.33 |
| CLT | 171–192 | 0.40 | 0.69 | ~0.34 |
| CLT + skip | 218 | **0.55** | 0.79 | ~0.28 |

The skip adds about +0.14 cos to both architectures, roughly what doubling L0 buys. Both stay equal
to each other. The skip also cuts the CLT's magnitude under-response (ratio 0.69 → 0.79). It is the
same lever that explains the Llama PLT's lead (its skip path alone scores 0.65).

## G. GPT-2 medium replication (24 layers, 4,096 features/layer, 150M tokens)
Adjacent layers:

| transcoder | recipe | L0 | cos | ratio | own-layer write |
|---|---|---|---|---|---|
| ridge linear floor | – | – | 0.32 | 0.41 | – |
| PLT | fixed penalty λ=0.01 | 247 | 0.25 | 0.57 | 1 |
| PLT | L0 controller | 465 | 0.31 | 0.63 | 1 |
| CLT | L0 controller | 476 | **0.35** | 0.53 | 0.61 |
| CLT | fixed penalty λ=0.01 | 135 | 0.14 | 0.53 | 0.27 |

- **At matched L0 the CLT is not behind:** 0.35 vs 0.31 at L0 ≈ 470. This replicates the GPT-2
  small result in a model twice as deep.
- The penalty-trained CLT writes 73% of its energy across layers, as heavily cross-layer as the
  public CLTs. It lands at 0.14, about 0.05 below the PLT line extrapolated from two points
  (predicted 0.19 at L0 135). That gap is inside the extrapolation's uncertainty. It is
  consistent with "on the curve" and leaves open a small penalty-CLT deficit at this depth.
- Medium transcoders at this width are weak overall: both PLTs sit at or below the linear floor
  at adjacent layers. A wider dictionary would be needed for a sharper medium-scale test.
- The CLT magnitude under-response replicates: ratio 0.53 vs 0.63.

## Conclusion
1. A causal instrument for replacement-model faithfulness: bit-exact frozen-error hybrid,
   resample-ablate MLP_s, compare every later MLP's response. It is cheap, needs no labels, and
   carries POS/NEG/linear controls.
2. In controlled conditions (GPT-2 small, GPT-2 medium) and in the public Gemma pair, the CLT
   deficit is sparsity: no architecture effect once L0 is accounted for. The public Llama CLT is
   the exception: a large deficit (−0.42 at matched L0, below a linear map) that sparsity and skip
   connections do not explain. That remains open.
3. **Any CLT-vs-PLT faithfulness comparison must match L0** (and skip connections). That includes
   the Gemma-2 CLT-vs-GemmaScope comparison used as evidence in Lange et al.'s post: the two have
   very different per-layer L0 profiles.
4. The sum-of-norms decoder penalty never helps, and under a fixed penalty it makes things worse.
5. Skip connections are the cheap lever: +0.14 for either architecture. CLT + skip at L0 218
   (0.55) is as faithful as a plain transcoder would be at roughly twice the L0.
6. GPT-2 medium replicates the matched-L0 result (CLT 0.35 vs PLT 0.31). A small deficit for
   heavily cross-layer, penalty-trained CLTs at depth can't be ruled out from these four runs.
