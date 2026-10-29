# Nudge

**An automated lab for discovering, testing and explaining neuron-steering interventions in language models.**

Give Nudge a behavioral target ("sound like a secret-society thriller", "express uncertainty when evidence is
insufficient"). It turns the target into target/contrast descriptions, samples candidate MLP neurons, steers
them during generation, scores what changed, compares each candidate against matched random-direction
controls, re-tests finalists on validation prompts, **freezes** the shortlist, evaluates it **once** on held-out
prompts, and writes an auditable report.

> *Use the agent to automate experiments, not to automate certainty.*
> A leaderboard score is a lead. Controls, held-out tests and reading the raw outputs decide whether it matters.



## Quick start

```bash
pip install -e ".[hf,ui,dev]"      # add ",embed" for sentence-transformers embeddings
nudge doctor                       # Phase-0 wiring checks (hook equivalence, cleanup, controls)
nudge demo                         # full pipeline on the random tiny model (~1 min, CPU)
nudge ui                           # browse runs, leaderboard, side-by-side outputs, report
pytest                             # 45 tests
```

Real model:

```bash
pip install -e ".[hf,embed,ui]"
nudge doctor --backend hf --model Qwen/Qwen2.5-0.5B-Instruct   # MUST pass before you trust anything
nudge plan configs/uncertainty_hf.toml                          # generation count before launching
nudge run  configs/uncertainty_hf.toml                          # resumable; re-run to continue
nudge report runs/uncertainty-qwen
```

`nudge run` prints a leaderboard, runs validation, freezes finalists, evaluates the held-out set once, and
writes `runs/<name>/report.md`. Stages can be run separately: `--stage screen|validate|test`.

## How it works

```
target text ─► TargetSpec (target/contrast descriptions, guardrails; versioned, human-editable)
                    │
 seeded sampler ─► candidate (layer, neuron) × doses ──► steered generations ─┐
 random-direction controls (same layer, dose, magnitude) ─────────────────────┤  stored one record each
 baselines (same prompts, same decoding, same seed) ──────────────────────────┘  (generations.jsonl)
                    │
        scoring (embedding J = S_T − S_C, coherence, behavioral evaluator)   (scores.jsonl, re-scorable)
                    │
 ΔJ vs baseline ─► leaderboard (+ control percentile/p, BH q-value, coherence, per-prompt variability)
                    │  shortlist: positive, coherent, above controls, behaviorally diverse
 validation prompts ─► refine dose ─► FREEZE finalists + success criteria (finalists.json)
                    │
 held-out test prompts (evaluated once) ─► confirmed / not_confirmed / inconclusive ─► report.md
```

The steering intervention adds `dose · R_layer · unit(W_down[:, i])` to the MLP output of the chosen layer at every position
(`R_layer` = median per-token L2 norm of that layer's MLP output on calibration text), which is equivalent to pushing
neuron *i*'s activation. Details, caveats and the exact statistics are in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Scientific safeguards (enforced in code, not just documented)

- **Frozen finalists and one look at the test set.** `test` refuses to run before `finalists.json` exists, refuses a
  second run, and a forced re-run permanently stamps the report *contaminated*.
- **Split integrity.** Screening/validation/held-out prompts are checked for id and text overlap at load time.
- **Matched random controls** per (layer, dose), seeded, same output magnitude; percentile + empirical p + Benjamini–Hochberg q across all screened cells.
- **Hooks cannot leak.** `try/finally` context managers; tests prove a baseline after a steered run is unchanged, even after exceptions.
- **Self-checks before experiments** (`nudge doctor`): column-vs-row orientation, output-hook ≡ activation-hook equivalence, zero-dose identity, cleanup, repeatability.
- **Guards against invalid hook points:** refuses blocks with post-FFN norm/layer-scale and quantized `down_proj`.
- **Auditability:** every generation stores model + weights fingerprint, tokenizer, layer/neuron, dose **and unit**, prompt, decoding config, seed, output, status. Scores are versioned by `scorer_id` (spec hash + embedder + formula + evaluators), so nothing changes silently.
- **Honest language in reports:** exploratory vs confirmed are separate sections; "steering X changed the output under these conditions", never "neuron X is the concept".

## Layout

```
configs/         demo_tiny.toml, uncertainty_hf.toml, thriller_hf.toml
data/prompts/    fiction_scenes.jsonl, uncertainty_qa.jsonl   (12 screening / 12 validation / 20 held-out each)
docs/            METHODOLOGY.md, DESIGN.md, example_report_tiny.md
src/nudge/
  targets/       schema.py (TargetSpec), target_builder.py (presets, LLM-proposed specs)
  models/        base.py (SteeringBackend), hf_backend.py, tiny.py (test fixture), explorer_adapter.py (stub)
  steering/      interventions.py, hooks.py, validation.py (nudge doctor)
  experiments/   prompts.py, sampling.py, storage.py, runner.py, confirm.py, config.py
  scoring/       embeddings.py, semantic.py, behavioral.py, judges.py, controls.py, ranking.py, diversity.py, pipeline.py
  reports/       markdown_report.py
  ui/app.py      Streamlit
  cli.py         `nudge` command
tests/           hook cleanup, shapes/orientation, reproducibility, scoring, split integrity, pipeline, UI
```

## Choosing and extending things

- **Target:** `nudge target --list`, `--preset NAME --out spec.json`, then edit the JSON (or use the UI). `propose_spec(goal, llm=callable)` lets any LLM *propose* descriptions; they are flagged `reviewed=False` until a human edits them.
- **Prompts:** JSONL with `id, text, partition, tags, meta`. For unpartitioned lists use `split_flat_list`.
- **Behavioral evaluator:** implement `Evaluator.score(ctx) -> dict`; `LLMJudge` takes any `str -> str` callable (blinded prompt, raw reply retained, randomized A/B pairwise helper).
- **Backend:** implement `SteeringBackend` (7 methods) and validate with the checks in `steering/validation.py`.

## Known limitations

1. **Not yet run on a pretrained model.** The build environment had no model download access. All 45 tests exercise the real hooks on a randomly-initialized tiny Llama; behavior on, e.g., Qwen2.5 is expected to work but unproven. Run `nudge doctor --backend hf --model …` first and treat its output as the gate.
2. **The tiny demo produces gibberish** and "finds" nothing. That is correct: on the demo run, screening winners fail confirmation, which is the selection bias the controls and held-out set exist to expose.
3. **Default embedder `hashing` is a lexical baseline,** used for offline tests. Use `sentence-transformers` for real runs and review J against raw outputs.
4. **Heuristic coherence and the lexicon uncertainty evaluator are proxies.** Validate any LLM judge on a manually-labelled sample before relying on it.
5. **Small default held-out sets (20 prompts)** give wide CIs; the criteria are defaults, not a statistical guarantee. Expand for noisy effects and report the search budget.
6. **Doses are not portable** across models or dose units; early layers (<3) behave badly under relative doses.
7. **Built-in prompt suites load from the repo checkout** (`data/prompts`); a non-editable wheel install should pass an explicit path.
8. Throughput is one generation at a time (no batching, no vLLM). Fine for ~10⁴ short generations on a GPU; large sweeps want a faster backend (`explorer_adapter.py` documents where).

## Attribution

Inspired by [Neuron Explorer](https://github.com/lyramakesmusic/neuron-explorer). **No upstream code is included**; the upstream repo ships no license, so Nudge reimplements the ideas independently. See [`NOTICE.md`](NOTICE.md). Nudge itself is MIT-licensed.
