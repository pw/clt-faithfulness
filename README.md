# clt-faithfulness

A causal test of whether cross-layer transcoders (CLTs) skip a language model's real
computation, the hypothesis of Lange et al., "Cross-Layer Transcoders are incentivized to learn
Unfaithful Circuits" (ICML 2026).

**Write-up:** https://byclaude.net/research/clt-faithfulness (charts: https://byclaude.net/research/clt-faithfulness/charts) · full tables in
[`results/SUMMARY.md`](results/SUMMARY.md)

## Result in one paragraph

On the public Llama-3.2-1B and Gemma-2-2B transcoders, CLTs track the real model's response to
mid-layer interventions much worse than per-layer transcoders (PLTs). That gap does not come from
cross-layer routing. On Gemma it is sparsity: at matched per-layer L0 the two are equal. On Llama
it is the PLT's linear skip connection. In a controlled sweep of 28 GPT-2-small transcoders, CLTs
and PLTs fall on one faithfulness-vs-L0 curve, whatever their training recipe and with
cross-layer write shares from 18% to 85%. The proposed sum-of-norms fix is the one variant below
the curve. A learned skip connection adds about +0.14 to either kind. A four-run GPT-2-medium
replication agrees at matched L0.

## The instrument

For each source layer *s*, resample-ablate MLP<sub>s</sub>'s output: replace it with its value
on a different held-out text. Then compare how every later MLP output responds in

- the real model, and
- the replacement model: every MLP replaced by transcoder reconstruction plus the clean-run error,
  held frozen. It is written in delta form, so it is bit-exact with the real model on clean input
  (asserted, KL < 1e-4).

Metrics: token-level cosine of the two responses (direction) and their norm ratio (magnitude).
Controls run every time: real MLPs (must give 1), frozen MLPs (must give 0), and a ridge
linear map (the floor).

## Layout

| path | what |
|---|---|
| `cltf/gpt2.py` | plain GPT-2 forward pass, checked against HuggingFace logits |
| `cltf/transcoder.py` | one class for PLT / CLT / windowed CLT, concat or sum-of-norms penalty, optional skip |
| `cltf/train.py` | online training from an activation shuffle buffer; L0 controller or fixed penalty |
| `cltf/evaluate.py` | the intervention instrument for GPT-2 (+ controls, CLT-charitable "write+source" variant) |
| `cltf/llama.py`, `cltf/gemma.py` | the same instrument on the public circuit-tracer transcoders |
| `cltf/writeshare.py`, `cltf/l0profile.py` | per-layer L0 and the share of write energy kept in each feature's own layer |
| `scripts/run_ctrl.sh`, `scripts/run_fixed.sh` | train → evaluate → measure, one transcoder per H100 |

Compute: about $400 of single-H100 time. Each GPT-2-small transcoder (300M FineWeb-edu tokens,
8,192 features per layer) trains in 1.8–3 hours.

Per-run result JSONs for the public models are in `results/`. The GPT-2 per-run JSONs are
summarized in `SUMMARY.md` and are available on request.

Written and run by Claude (Anthropic's model) working with Patrick White.
Questions or a flaw: me@byclaude.net.
