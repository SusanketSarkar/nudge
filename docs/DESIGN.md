# Design notes

**Why append-only JSONL?** One structured record per generation is trivially diffable, resumable (truncated last lines are skipped), and re-scorable. A DB server was explicitly out of scope; SQLite is an easy later swap behind `RunStore`.

**Why scoring is separate from generation.** Embedders, target descriptions and evaluators change; generations are expensive. Scores are keyed `(generation key, scorer_id)`, so re-scoring never touches rollouts and old scores remain.

**Why a `SteeringBackend` ABC.** Everything above it (sampler, runner, controls, UI) is model-agnostic. `HFBackend` is the reference implementation; the tiny random Llama (`models/tiny.py`) is a hermetic fixture so CI and demos need no downloads.

**Why dose = fraction of layer MLP-output norm.** It makes candidates in a layer comparable and makes random controls trivial to match. Raw activation units are available for equivalence checks.

**Deviations from the original spec's tree.** Confirmation logic lives in `experiments/confirm.py` (not `steering/validation.py`, which hosts the Phase-0 wiring checks). The agent-driven planner (P2) is not included; the runner exposes the stage methods an agent would call (`screen`, `shortlist`, `validate`, `freeze`, `test`).

**Not implemented (by design for the MVP):** pair/combination search, activation prescreening, steering-lens/logit-lens, 3D map, MCP tools, vLLM path, batching. The UI's "Test combination" button is present but disabled.

**Next steps worth doing first:** (1) run `nudge doctor` and a small pilot on a real ≤1.5B model; (2) swap in sentence-transformers and inspect raw J vs. outputs; (3) add an LLM judge validated on ~50 hand-labelled outputs; (4) raise control count to ≥20 and held-out to ≥40 prompts; (5) replicate on a second checkpoint.
