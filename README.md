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

### 1. Install

```bash
pip install -e ".[hf,embed,ui,dev]"   # torch/transformers, sentence-transformers, streamlit, pytest/ruff
```

Drop `embed` (`".[hf,ui,dev]"`) if you only need the offline `hashing` embedder and want to skip the
sentence-transformers download. This only prepares the environment; no experiment runs yet.

### 2. Check the wiring before trusting anything

```bash
nudge doctor                                                    # tiny random model, no download
nudge doctor --backend hf --model Qwen/Qwen2.5-3B-Instruct    # the model you actually plan to use
```

`doctor` verifies that the steering hooks are installed at the right place and removed again: column-vs-row
orientation, output-hook vs activation-hook equivalence, zero-dose identity, cleanup and repeatability. It also
prints `n_layers` and `intermediate_size`, which you need later for a full-neuron sweep. **Do not run experiments
until it says `all checks passed`.**

### 3. Try the pipeline once (optional, ~1 min on a CPU)

```bash
nudge demo
```

Runs the whole screen → validate → freeze → test → report pipeline on a randomly initialized tiny model. Outputs
are gibberish and it finds nothing: the point is to prove the plumbing works before spending GPU time.

### 4. Pick or create a target

A target is a JSON `TargetSpec` (target/contrast descriptions, guardrails). Use a built-in preset:

```bash
nudge target --list
nudge target --preset secret_society_thriller --out configs/my_target.json
```

Or generate a skeleton for a new behavior, then edit the file:

```bash
nudge target --goal "make the LLM sound like a zombie" --out configs/my_target.json
```

The skeleton is flagged `reviewed: false` and lints as untrusted: write real target/contrast descriptions and set
`"reviewed": true`.

To have an LLM draft the descriptions instead, pass `--auto`. It talks to any OpenAI-compatible chat endpoint —
a local vLLM server, llama.cpp, or a hosted API — and writes the proposed JSON for you to edit:

```bash
nudge target --auto --goal "make the LLM sound like a zombie" --out configs/my_target.json
# vLLM on another host / model:
nudge target --auto --goal "make the LLM sound like a zombie" --out configs/my_target.json \
  --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct
```

`--base-url` defaults to `$NUDGE_LLM_BASE_URL` or `http://localhost:8000/v1`; `--model` defaults to
`$NUDGE_LLM_MODEL` and otherwise the first model the server advertises; `--api-key` reads
`$NUDGE_LLM_API_KEY` / `$OPENAI_API_KEY`. The output is still `origin: "llm_proposed"`, `reviewed: false` —
[IMP] the LLM only proposes, you have to check and edit and mark it reviewed before the run will trust it.

### 5. Write a config

Copy-paste a template (`configs/uncertainty_hf.toml`, `configs/thriller_hf.toml`), point `target` at your spec, and set your prompt suite:

```toml
[experiment]
name = "zombie-qwen"
target = "configs/my_target.json"   # preset name or path to spec .json
run_dir = "runs"

[backend]
kind = "hf"
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
min_layer = 3

[search]
survey_layers = 5        # layers spread across depth (or list explicit `layers = [...]`, or set it to -1 to run for all layers)
neurons_per_layer = 20   # random sample per layer; you can set this to -1 as well and seep through all neurosn
doses = [0.5, 0.8]       # strength as a fraction of the layer's median MLP-output norm

[prompts]
suite = "fiction_scenes" # built-in suite or path to a .jsonl with screening/validation/test partitions

[scoring]
embedder = "sentence-transformers/all-MiniLM-L6-v2"
```

Every field and its default is in `src/nudge/experiments/config.py`; the `[selection]` and `[criteria]` sections
control shortlist size, coherence floor and what counts as a confirmed result.

### 6. Estimate cost, then run

```bash
nudge plan configs/zombie_hf.toml                  # generation count and cached cells; run this first
nudge run  configs/zombie_hf.toml                  # full pipeline; resumable — re-run to continue
nudge run  configs/zombie_hf.toml --stage screen   # or just screening (leaderboard only)
nudge report runs/zombie-qwen                      # write runs/<name>/report.md
nudge ui --runs runs                               # browse leaderboard, per-prompt outputs, reports
```

`nudge run` screens candidates against baselines and matched random-direction controls, shortlists the credible
ones, re-tests them on validation prompts, freezes finalists and success criteria, then evaluates the held-out set
**once**. Stages can be run separately with `--stage screen|validate|test`; `validate --freeze` writes
`finalists.json` without testing. Because every cell is cached by model + prompt + intervention + decoding config,
interrupting and re-running continues where it stopped.

**Manual check.** To eyeball a leaderboard row on your own prompt, without launching a new experiment:

```bash
nudge check L12.N4254 --run runs/zombie-qwen --dose 0.8 --query "what is a dolphin?"
nudge check L12.N4254 --run runs/zombie-qwen --dose -0.8 --dose 0.8 \
  --query "what is a dolphin?" --query-file my_queries.txt --controls 16
```

It prints the baseline and steered answers side by side, the shift (`deltaJ`), target/contrast similarities,
coherence/relevance deltas, and optionally how the shift compares with matched random-direction controls
(`--controls N`; `-1` uses the config's `per_layer_dose`). Use `--config configs/zombie_hf.toml` instead of
`--run` if the run has no saved `spec.json`, and `--json` for machine-readable output.

### 7. Sweeping every neuron

By default `neurons_per_layer` is a random sample, not all neurons. Use `-1` to inspect **every** neuron in each
selected layer, and `survey_layers = -1` to survey **every** layer (still respecting `backend.min_layer`). You can
also scan one whole layer with `layers = [L]` + `neurons_per_layer = -1`.

```toml
[search]
survey_layers = -1         # all layers across depth (or an explicit `layers = [...]` list)
neurons_per_layer = -1     # every MLP neuron in each selected layer
doses = [0.7]              # one dose first; two doubles the cost
```

A full model-wide sweep is huge: candidate count is `layers × neurons_per_layer × doses × prompts` — hundreds of
thousands of generations on a 0.5B model. Use one dose and `n_screening = 1–2` first, start with `--stage screen`,
read the leaderboard, and only validate/test the top candidates.

### 8. Run the tests

```bash
pytest        # hooks, orientation, reproducibility, scoring, split integrity, pipeline, UI
ruff check src tests
```

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

## Scientific safeguards in code:

- **Frozen finalists and one look at the test set.** `test` refuses to run before `finalists.json` exists, refuses a
  second run, and a forced re-run permanently stamps the report *contaminated*.
- **Split integrity.** Screening/validation/held-out prompts are checked for id and text overlap at load time.
- **Matched random controls** per (layer, dose), seeded, same output magnitude; percentile + empirical p + Benjamini–Hochberg q across all screened cells.
- **Hooks cannot leak.** `try/finally` context managers; tests prove a baseline after a steered run is unchanged, even after exceptions.
- **Self-checks before experiments** (`nudge doctor`): column-vs-row orientation, output-hook ≡ activation-hook equivalence, zero-dose identity, cleanup, repeatability.
- **Guards against invalid hook points:** refuses blocks with post-FFN norm/layer-scale and quantized `down_proj`.
- **Auditability:** every generation stores model + weights fingerprint, tokenizer, layer/neuron, dose **and unit**, prompt, decoding config, seed, output, status. Scores are versioned by `scorer_id` (spec hash + embedder + formula + evaluators), so nothing changes silently.
- **Honest language in reports:** exploratory vs confirmed are separate sections; "steering X changed the output under these conditions", never "neuron X is the concept".



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
