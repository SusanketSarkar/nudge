# Methodology

## 1. Intervention

For a gated MLP, `y = W_down · h` with `h = act(W_gate x) ⊙ (W_up x)`. Pushing neuron *i* by α:
`y' = y + α · W_down[:, i]`, equivalent to `h_i → h_i + α`, applied at every position (prompt and generated).
No weights change. `nn.Linear` stores `W_down` as `[hidden, intermediate]`, so a neuron is a **column**.
`steering.validation.check_delta_shape_and_orientation` and `check_equivalence` verify this at runtime.

**When this is *not* a neuron push:** if anything non-linear sits between `down_proj` and the residual add
(Gemma-2/3 post-FFN norm, layer scale), adding at the `down_proj` output is a different intervention.
`HFBackend` refuses such blocks unless `allow_post_ffn_norm=True`. Quantized `down_proj` is refused as well.

### Dose units
| unit | definition |
|---|---|
| `raw_activation` | α = dose. Scale depends on the neuron and model. |
| `rel_mlp_out_norm` *(default)* | `delta = dose · R_layer · W_down[:, i]/‖W_down[:, i]‖`, `R_layer` = median per-token L2 norm of that layer's MLP output on 8 calibration texts (hash recorded in metadata). Every neuron in a layer receives the same push magnitude. |

Doses are comparable only within one model, one unit and one calibration. Early layers (<3) misbehave under relative doses;
`min_layer` defaults can skip them and the run metadata warns when they are used.

## 2. Candidate sampling
Uniform without replacement within each chosen layer, seeded by `(seed, layer)` so adding a layer does not reshuffle
others. Optional statistics-assisted mode stratifies by quantile bins of a per-neuron statistic (never top-k by activation:
a causally useful neuron need not be the loudest). Nothing assumes "concepts live in late layers".

## 3. Scoring
`J(y) = mean_t cos(E(y),E(t)) − mean_c cos(E(y),E(c))`; `ΔJ = J(steered) − J(baseline)` on the **same prompt, decoding config and seed**.
Complete outputs are embedded. Raw similarities are stored. J says an output moved toward a *description*; it is not a
calibrated probability and not evidence about what a neuron represents.

Coherence (heuristic, [0,1]): `printable_ratio · min(1, distinct_bigram_ratio/0.6)`, ×0.3 if the text ends in a repeated loop; flags `empty/loop/garbled/very_short`.
Relevance: cosine(prompt, output). Behavioral evaluators are pluggable; built-in `uncertainty_lexicon` uses a hedge-phrase lexicon plus answer aliases.

## 4. Random-direction controls
For each (layer, dose): K random unit vectors in the residual (MLP-output) space, scaled to the same magnitude as the candidate push
(for `rel_mlp_out_norm` that is `|dose|·R_layer`; for `raw_activation` the anchored neuron's `|dose|·‖W_down[:, i]‖`). Directions are seeded and deterministic.
For candidate with prompt set P, its mean ΔJ over `P ∩ control prompts` is compared with each control's mean ΔJ over the same prompts:

- **percentile** = fraction of controls strictly below the candidate;
- **p** = (1 + #{controls ≥ candidate}) / (K + 1) (one-sided; with K=8 the smallest possible p is 0.11, so use K ≥ 20 if you want 0.05-level evidence);
- **q** = Benjamini–Hochberg across all screened (neuron, dose) cells.

Control comparison uses *all* OK outputs on both sides (no coherence filtering), otherwise controls that destroy the text would be dropped and make candidates look better. Coherence is a separate guardrail on the candidate.

## 5. Selection → validation → freeze → test
1. Screening ranks best (sign, dose) per neuron by mean ΔJ among eligible candidates (≥ `min_valid_prompts` valid outputs, mean coherence ≥ `min_coherence`).
2. Shortlist: ΔJ > 0, above controls (if run), greedy diversity over per-prompt ΔJ profiles (cosine < `max_profile_cosine`). Clusters are an organizational aid, not proof of distinct mechanisms.
3. Validation re-tests shortlisted neurons on new prompts at all doses; finalists = best validation dose per neuron, positive and above controls.
4. `freeze` writes finalists + criteria + hashes. 5. `test` evaluates held-out prompts once.

### Confirmation criteria (defaults, preregistered in `finalists.json`)
`confirmed` requires all of: ≥10 valid held-out prompts; bootstrap 95% CI of mean ΔJ > 0 (resampling prompts); ΔJ>0 on ≥60% of prompts; mean Δcoherence ≥ −0.15;
control percentile ≥ 0.95; (if answerability data exist) answerable-question success drop ≤ 0.10. Too few prompts or no controls → `inconclusive`. Otherwise `not_confirmed` with the reasons listed.

## 6. What a "confirmed" candidate means
That steering that neuron at that dose moved ΔJ reliably on unseen prompts for this model, embedder, and decoding setup, more than matched random pushes, without a large coherence loss. It does **not** show that the neuron encodes the concept, that the effect generalizes to other checkpoints, or that J reflects human judgement. Read the examples; replicate.

## 7. Reproducibility record
Per generation: model id, weights fingerprint, tokenizer, layer, neuron, dose, dose unit, hook point, prompt, decoding config, seed, output, status/error, latency. Per run: spec (+hash), config, environment versions, prompt-suite fingerprint, search budget, scorer id. Caches are keyed by all of these, so edits never silently reuse stale cells.
