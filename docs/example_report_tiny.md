# Nudge experiment report: `secret_society_thriller`

*Generated 2026-10-28 03:16 UTC · run `demo-tiny` · target spec `64c56b7af6c3` · scorer `00237796b524`*

> ⚠️ Layers < 3 selected: relative doses are unreliable in the first layers (many empty/looping outputs, controls included); inspect coherence closely.

> ⚠️ Backend is the random tiny fixture: outputs are gibberish; results are plumbing tests only.

## 1. Research question and preregistered criteria

**Goal.** Make fictional prose sound like a secret-society conspiracy thriller.

**Question.** Can steering single MLP neurons move the model's outputs toward the target description (relative to contrast descriptions and to matched random directions) in a way that holds on held-out prompts?

**Target descriptions**

- A tense conspiracy thriller about a secret society, hidden symbols and whispered warnings.
- Shadowy figures in a hidden order meet in secret, and nobody can be trusted.
- The narrator realises they are being watched by an ancient secret brotherhood.

**Contrast descriptions**

- Ordinary everyday description of a mundane scene in plain, calm language.
- A cheerful, straightforward account of a normal day with nothing unusual in it.
- Neutral factual prose describing ordinary places and routine activities.

**Guardrails**

- Do not reward incoherent or irrelevant output.

**Confirmation criteria** (frozen before the held-out evaluation)

```json
{
  "min_prompts": 8,
  "ci_level": 0.95,
  "min_success_rate": 0.6,
  "max_coherence_drop": 0.15,
  "min_control_percentile": 0.5,
  "max_answerable_drop": 0.1,
  "bootstrap_seed": 0
}
```

## 2. Model and environment

| item | value |
|---|---|
| model_id | nudge-tiny-random-llama-seed0 |
| model_class | LlamaForCausalLM |
| tokenizer_id | nudge-byte-tokenizer |
| n_layers | 6 |
| hidden_size | 64 |
| intermediate_size | 192 |
| dtype | float32 |
| device | cpu |
| hook_point | mlp_out (additive, output of down_proj) |
| weights_fingerprint | d6705e2ebb79 |
| python | 3.12.3 |
| platform | Linux-6.18.44-fc-v180-x86_64-with-glibc2.39 |
| torch | 2.14.1+cu130 |
| transformers | 5.21.0 |
| numpy | 2.4.4 |
| pandas | 3.0.2 |

**Dose definitions (doses are comparable only within one unit and one model):**

- `raw_activation`: alpha added to the neuron's activation h_i (delta = alpha * W_down[:, i]).
- `rel_mlp_out_norm`: delta = dose * median_token_L2_norm(layer MLP output) * unit(W_down[:, i]); the same push magnitude for every neuron in a layer.

## 3. Search procedure and budget

- Candidate selection: seeded uniform random sampling within layers [2, 3, 4] (seed 0).
- Neurons per layer: 6 → **18 candidate neurons** × doses [-1.0, 1.0].
- Random-direction controls: 8 per (layer, dose); same layer, dose and output magnitude as the candidate, random direction in the MLP-output space.
- Prompt suite `fiction_scenes` (c3f8d3b783df), partitions: {'screening': 12, 'validation': 12, 'test': 20}.
- Generation config: `{"max_new_tokens": 24, "temperature": 0.0, "top_p": 1.0, "seed": 0, "use_chat_template": false, "system_prompt": null}`; identical for baseline, steered and control runs.
- Because many neurons were tested, some will look good by chance. p-values below are one-sided empirical (vs. random controls) and q-values are Benjamini–Hochberg adjusted across all screened (neuron, dose) cells.

## 4. Scoring

`J=mean_cos(target)-mean_cos(contrast);v1`. Embedder: `hashing-v1-d1024`. For output *y*, S_T = mean cosine to target descriptions, S_C = mean cosine to contrast descriptions, J = S_T − S_C, ΔJ = J(steered) − J(baseline, same prompt). Coherence is a heuristic in [0, 1] (distinct-bigram ratio, printable ratio, loop penalty). **J is an embedding-similarity signal, not a behavioural oracle and not evidence that a neuron represents the concept.**

## 5. Exploratory findings (screening)

36 (neuron, dose) cells screened; 36 eligible (≥ 4 valid prompts, mean coherence ≥ 0.0).

| rank | neuron | dose | n | mean ΔJ | sd | ΔJ>0 | coh. | Δcoh. | Δbehav. | ctrl pct | p | q |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | L2.N124 | 1.000 | 8 | 0.027 | 0.063 | 0.625 | 0.070 | -0.042 | – | 1.000 | 0.111 | 0.727 |
| 2 | L2.N85 | 1.000 | 8 | 0.024 | 0.029 | 0.625 | 0.350 | 0.238 | – | 1.000 | 0.111 | 0.727 |
| 3 | L3.N24 | 1.000 | 8 | 0.023 | 0.027 | 0.625 | 0.326 | 0.214 | – | 0.875 | 0.222 | 0.727 |
| 4 | L3.N177 | 1.000 | 8 | 0.020 | 0.034 | 0.500 | 0.104 | -0.008 | – | 0.875 | 0.222 | 0.727 |
| 5 | L3.N45 | 1.000 | 8 | 0.015 | 0.029 | 0.375 | 0.072 | -0.040 | – | 0.875 | 0.222 | 0.727 |
| 6 | L2.N50 | -1.000 | 8 | 0.015 | 0.045 | 0.500 | 0.297 | 0.185 | – | 0.875 | 0.222 | 0.727 |
| 7 | L4.N0 | -1.000 | 8 | 0.013 | 0.022 | 0.625 | 0.087 | -0.025 | – | 0.875 | 0.222 | 0.727 |
| 8 | L2.N22 | -1.000 | 8 | 0.012 | 0.034 | 0.250 | 0.081 | -0.031 | – | 0.750 | 0.333 | 0.923 |
| 9 | L4.N126 | -1.000 | 8 | 0.011 | 0.022 | 0.625 | 0.123 | 0.011 | – | 0.875 | 0.222 | 0.727 |
| 10 | L3.N19 | 1.000 | 8 | 0.010 | 0.024 | 0.625 | 0.139 | 0.027 | – | 0.625 | 0.444 | 1.000 |
| 11 | L2.N132 | -1.000 | 8 | 0.009 | 0.035 | 0.500 | 0.105 | -0.007 | – | 0.500 | 0.556 | 1.000 |
| 12 | L4.N50 | 1.000 | 8 | 0.008 | 0.023 | 0.375 | 0.089 | -0.023 | – | 0.875 | 0.222 | 0.727 |
| 13 | L4.N82 | 1.000 | 8 | 0.008 | 0.021 | 0.500 | 0.081 | -0.031 | – | 0.875 | 0.222 | 0.727 |
| 14 | L4.N164 | -1.000 | 8 | 0.007 | 0.024 | 0.375 | 0.083 | -0.029 | – | 0.750 | 0.333 | 0.923 |
| 15 | L3.N90 | 1.000 | 8 | 0.004 | 0.039 | 0.250 | 0.053 | -0.059 | – | 0.125 | 0.889 | 1.000 |

These are **leads**, not findings: they were selected from the same prompts they are scored on.

### Example outputs for the top screening candidate (L2.N124, dose +1)

- **Prompt** `fic-scr-06` (ΔJ = 0.157)  
  *baseline:* �������y�y�y�y�y�  
  *steered:* �����������!�j
- **Prompt** `fic-scr-02` (ΔJ = 0.072)  
  *baseline:* ���������y�y�y�  
  *steered:* ����������!�j�y

## 6. Validation

| rank | neuron | dose | n | mean ΔJ | sd | ΔJ>0 | coh. | Δcoh. | Δbehav. | ctrl pct | p | q |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | L2.N85 | 1.000 | 8 | 0.023 | 0.029 | 0.500 | 0.287 | 0.260 | – | 0.875 | 0.222 | 0.356 |
| 2 | L3.N24 | 1.000 | 8 | 0.020 | 0.029 | 0.625 | 0.314 | 0.288 | – | 1.000 | 0.111 | 0.356 |
| 3 | L2.N124 | 1.000 | 8 | 0.014 | 0.035 | 0.625 | 0.074 | 0.048 | – | 0.625 | 0.444 | 0.593 |
| 4 | L3.N177 | -1.000 | 8 | 0.012 | 0.027 | 0.625 | 0.054 | 0.028 | – | 0.875 | 0.222 | 0.356 |

## 7. Held-out confirmation

Finalists frozen before testing (sha `ddc08b7d5328898b`). **0 of 2 finalists confirmed.**

| neuron | dose | status | n | mean ΔJ | CI low | CI high | ΔJ>0 | Δcoh. | ctrl pct |
|---|---|---|---|---|---|---|---|---|---|
| L2.N85 | 1.000 | not_confirmed | 12 | -0.003 | -0.018 | 0.011 | 0.417 | 0.045 | 0.500 |
| L3.N24 | 1.000 | not_confirmed | 12 | -0.012 | -0.033 | 0.008 | 0.500 | 0.113 | 0.000 |

- `L2.N85` not_confirmed: CI for mean ΔJ includes 0 (-0.0185, +0.0114); success rate 0.42 < 0.6
- `L3.N24` not_confirmed: CI for mean ΔJ includes 0 (-0.0329, +0.0084); success rate 0.50 < 0.6; effect only at the 0.00 percentile of random controls (< 0.5)

### Held-out examples: L2.N85 (dose +1)

- **Prompt** `fic-tes-10` (ΔJ = 0.055)  
  *baseline:* y�y�y�y�y�y�y�y�y�y�y�y�  
  *steered:* yyyyyyyyyyyyyy����������
- **Prompt** `fic-tes-01` (ΔJ = 0.006)  
  *baseline:* �y]�y]�y����������������  
  *steered:* �y]�y]�y]��y]��y]��y]��y

### Held-out examples: L3.N24 (dose +1)

- **Prompt** `fic-tes-10` (ΔJ = 0.055)  
  *baseline:* y�y�y�y�y�y�y�y�y�y�y�y�  
  *steered:* yyyyyyyyyyyyyyyy��������
- **Prompt** `fic-tes-01` (ΔJ = 0.030)  
  *baseline:* �y]�y]�y����������������  
  *steered:* �yyy]�y]�y]�y]�y]M;y]M;y

## 8. Failures and negative results

- Failed/unscored generations: 0 of 1172.
- Steered outputs flagged loop/empty/garbled: 374 of 376 (99%).
- Finalists not confirmed on held-out prompts: 2 of 2 (reasons above).
- All candidates, including those that failed, remain in `generations.jsonl` and `scores.jsonl`.

## 9. What this does and does not establish

- Steering a neuron *changed the outputs* in the stated way **under these conditions** (model, prompts, dose, decoding). It does not show the neuron *represents* the concept; effects may be distributed or flow through other components.
- Embedding similarity and lexical evaluators are proxies; read the raw examples.
- Held-out prompts are few; confidence intervals are wide for small *n*. Replicate on more prompts, seeds and a second checkpoint.
- Neuron indices and doses do not transfer between checkpoints or dose units.

## 10. Reproduce

```bash
nudge run <this run's config.toml>      # resumes from the stored generations
nudge report runs/demo-tiny
```

Run directory contents: `spec.json`, `config.json`, `meta.json`, `generations.jsonl`, `scores.jsonl`, `finalists.json`, `confirmation.json`.