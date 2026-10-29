"""Self-contained Markdown report. Exploratory and confirmed findings are kept visibly apart."""
from __future__ import annotations

import datetime as dt
import json
from typing import Any

import numpy as np
import pandas as pd

from nudge.experiments.storage import COND_BASELINE, COND_STEERED, RunStore
from nudge.scoring import ranking as rk
from nudge.scoring.semantic import FORMULA_VERSION
from nudge.targets.schema import TargetSpec


def _f(x: Any, nd: int = 3) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.{nd}f}" if isinstance(x, (float, np.floating)) else str(x))


def _table(df: pd.DataFrame, cols: dict[str, str]) -> str:
    cols = {c: h for c, h in cols.items() if c in df.columns}
    lines = ["| " + " | ".join(cols.values()) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples():
        lines.append("| " + " | ".join(_f(getattr(r, c)) for c in cols) + " |")
    return "\n".join(lines)


def _md_escape(s: str) -> str:
    s = "".join(ch if (ch.isprintable() or ch == "\n") else "�" for ch in s)
    return s.replace("|", "\\|").replace("\n", " ⏎ ")


def _examples(rows: pd.DataFrame, neuron_id: str, dose: float, partition: str, n: int = 2) -> str:
    layer, neuron = (int(x) for x in neuron_id[1:].split(".N"))
    st = rows[(rows.partition == partition) & (rows.condition == COND_STEERED) & (rows.layer == layer)
              & (rows.neuron == neuron) & (rows.dose == dose) & rows.ok]
    out = []
    for r in st.sort_values("dj", ascending=False).head(n).itertuples():
        base = rows[rows.key == r.baseline_key]
        b = base.output.iloc[0] if len(base) else ""
        out.append(f"- **Prompt** `{r.prompt_id}` (ΔJ = {_f(r.dj)})  \n  *baseline:* {_md_escape(b[:300])}  \n  "
                   f"*steered:* {_md_escape(r.output[:300])}")
    return "\n".join(out)


def build_report(store: RunStore, scorer_id: str | None = None) -> str:
    meta = store.read_json("meta.json", {})
    scorer_id = scorer_id or meta.get("scorer_id") or (store.scorer_ids() or [None])[-1]
    spec = TargetSpec.load(store.path("spec.json"))
    cfg = store.read_json("config.json", {})
    rows = rk.build_rows(store, scorer_id)
    p = cfg.get("selection", {})
    params = rk.RankingParams(min_valid_prompts=p.get("min_valid_prompts", 5), min_coherence=p.get("min_coherence", 0.0),
                              control_percentile=p.get("control_percentile", 0.95))
    conf = store.read_json("confirmation.json")
    fin = store.read_json("finalists.json")
    model = meta.get("model", {})
    L: list[str] = []
    a = L.append

    a(f"# Nudge experiment report: `{spec.target_name}`\n")
    a(f"*Generated {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M UTC} · run `{store.root.name}` · "
      f"target spec `{spec.spec_hash}` · scorer `{scorer_id}`*\n")
    for w in meta.get("warnings", []):
        a(f"> ⚠️ {w}\n")
    if conf and conf.get("contaminated"):
        a("> ⚠️ **The held-out set was evaluated more than once for this run; the confirmatory results below are contaminated.**\n")

    a("## 1. Research question and preregistered criteria\n")
    a(f"**Goal.** {spec.goal or '(not recorded)'}\n")
    a("**Question.** Can steering single MLP neurons move the model's outputs toward the target description "
      "(relative to contrast descriptions and to matched random directions) in a way that holds on held-out prompts?\n")
    a("**Target descriptions**\n\n" + "\n".join(f"- {d}" for d in spec.target_descriptions) + "\n")
    a("**Contrast descriptions**\n\n" + "\n".join(f"- {d}" for d in spec.contrast_descriptions) + "\n")
    if spec.guardrails:
        a("**Guardrails**\n\n" + "\n".join(f"- {g}" for g in spec.guardrails) + "\n")
    crit = (fin or {}).get("criteria") or cfg.get("criteria", {})
    frozen = " (frozen before the held-out evaluation)" if fin else " (from config; not yet frozen)"
    a(f"**Confirmation criteria**{frozen}\n\n```json\n{json.dumps(crit, indent=2)}\n```\n")
    if spec.lint():
        a("**Specification lint:** " + "; ".join(spec.lint()) + "\n")

    a("## 2. Model and environment\n")
    a("| item | value |\n|---|---|")
    for k in ("model_id", "model_class", "tokenizer_id", "n_layers", "hidden_size", "intermediate_size", "dtype",
              "device", "hook_point", "weights_fingerprint"):
        a(f"| {k} | {model.get(k, '–')} |")
    for k, v in (meta.get("environment") or {}).items():
        a(f"| {k} | {v} |")
    a("")
    a("**Dose definitions (doses are comparable only within one unit and one model):**\n")
    for u, d in (model.get("dose_units") or {}).items():
        a(f"- `{u}`: {d}")
    a("")

    sb = meta.get("search_budget", {})
    a("## 3. Search procedure and budget\n")
    a(f"- Candidate selection: seeded uniform random sampling within layers {sb.get('layers')} (seed {cfg.get('seed')}).")
    a(f"- Neurons per layer: {sb.get('neurons_per_layer')} → **{sb.get('n_candidate_neurons')} candidate neurons** × doses {sb.get('doses')}.")
    a(f"- Random-direction controls: {sb.get('controls_per_layer_dose')} per (layer, dose); same layer, dose and "
      "output magnitude as the candidate, random direction in the MLP-output space.")
    a(f"- Prompt suite `{meta.get('prompt_suite')}` ({meta.get('prompt_fingerprint')}), partitions: {meta.get('prompt_counts')}.")
    a(f"- Generation config: `{json.dumps(cfg.get('generation', {}))}`; identical for baseline, steered and control runs.")
    a("- Because many neurons were tested, some will look good by chance. p-values below are one-sided empirical "
      "(vs. random controls) and q-values are Benjamini–Hochberg adjusted across all screened (neuron, dose) cells.\n")

    a("## 4. Scoring\n")
    a(f"`{FORMULA_VERSION}`. Embedder: `{meta.get('embedder')}`. For output *y*, "
      "S_T = mean cosine to target descriptions, S_C = mean cosine to contrast descriptions, J = S_T − S_C, "
      "ΔJ = J(steered) − J(baseline, same prompt). Coherence is a heuristic in [0, 1] (distinct-bigram ratio, "
      "printable ratio, loop penalty). **J is an embedding-similarity signal, not a behavioural oracle and not "
      "evidence that a neuron represents the concept.**\n")

    a("## 5. Exploratory findings (screening)\n")
    table = rk.candidate_table(rows, params, "screening")
    lb = rk.leaderboard(table, params)
    n_cells = len(table)
    a(f"{n_cells} (neuron, dose) cells screened; {int(table.eligible.sum()) if n_cells else 0} eligible "
      f"(≥ {params.min_valid_prompts} valid prompts, mean coherence ≥ {params.min_coherence}).\n")
    cols = {"rank": "rank", "neuron_id": "neuron", "dose": "dose", "n_prompts": "n", "mean_dj": "mean ΔJ",
            "std_dj": "sd", "success_rate": "ΔJ>0", "mean_coherence": "coh.", "d_coherence": "Δcoh.",
            "behavior_delta": "Δbehav.", "control_percentile": "ctrl pct", "p_control": "p", "q_value": "q"}
    if lb.empty:
        a("_No eligible candidates._\n")
    else:
        a(_table(lb.head(15), cols) + "\n")
        a("These are **leads**, not findings: they were selected from the same prompts they are scored on.\n")
        top = lb.iloc[0]
        a(f"### Example outputs for the top screening candidate ({top.neuron_id}, dose {top.dose:+g})\n")
        a(_examples(rows, top.neuron_id, top.dose, "screening") + "\n")

    a("## 6. Validation\n")
    vt = rk.candidate_table(rows, params, "validation")
    vlb = rk.leaderboard(vt, params)
    if vlb.empty:
        a("_No validation run._\n")
    else:
        a(_table(vlb, cols) + "\n")

    a("## 7. Held-out confirmation\n")
    if not conf:
        a("_The held-out test set has not been evaluated._ Nothing in this report is confirmed.\n")
    else:
        a(f"Finalists frozen before testing (sha `{conf['finalists_sha']}`). "
          f"**{conf['n_confirmed']} of {conf['n_finalists']} finalists confirmed.**\n")
        ccols = {"neuron_id": "neuron", "dose": "dose", "status": "status", "n_prompts": "n", "mean_dj": "mean ΔJ",
                 "ci_low": "CI low", "ci_high": "CI high", "success_rate": "ΔJ>0", "d_coherence": "Δcoh.",
                 "control_percentile": "ctrl pct", "behavior_delta_answerable": "Δ answerable",
                 "behavior_delta_unanswerable": "Δ unanswerable"}
        a(_table(pd.DataFrame(conf["candidates"]), ccols) + "\n")
        for c in conf["candidates"]:
            if c["reasons"]:
                a(f"- `{c['neuron_id']}` {c['status']}: " + "; ".join(c["reasons"]))
        a("")
        for c in conf["candidates"]:
            a(f"### Held-out examples: {c['neuron_id']} (dose {c['dose']:+g})\n")
            a(_examples(rows, c["neuron_id"], c["dose"], "test") + "\n")

    a("## 8. Failures and negative results\n")
    allc = rows[(rows.condition == COND_STEERED) & (rows.n_iv == 1)]
    n_err = int((~rows.ok).sum()) if len(rows) else 0
    a(f"- Failed/unscored generations: {n_err} of {len(rows)}.")
    if len(allc):
        flagged = allc[allc["flags"].str.contains("loop|empty|garbled", na=False)]
        a(f"- Steered outputs flagged loop/empty/garbled: {len(flagged)} of {len(allc)} "
          f"({100 * len(flagged) / len(allc):.0f}%).")
    if not lb.empty and conf:
        failed = [c for c in conf["candidates"] if c["status"] != "confirmed"]
        a(f"- Finalists not confirmed on held-out prompts: {len(failed)} of {len(conf['candidates'])} (reasons above).")
    a("- All candidates, including those that failed, remain in `generations.jsonl` and `scores.jsonl`.\n")

    a("## 9. What this does and does not establish\n")
    a("- Steering a neuron *changed the outputs* in the stated way **under these conditions** (model, prompts, dose, decoding). "
      "It does not show the neuron *represents* the concept; effects may be distributed or flow through other components.")
    a("- Embedding similarity and lexical evaluators are proxies; read the raw examples.")
    a("- Held-out prompts are few; confidence intervals are wide for small *n*. Replicate on more prompts, seeds and a second checkpoint.")
    a("- Neuron indices and doses do not transfer between checkpoints or dose units.\n")

    a("## 10. Reproduce\n")
    a("```bash\nnudge run <this run's config.toml>      # resumes from the stored generations\n"
      f"nudge report runs/{store.root.name}\n```\n")
    a("Run directory contents: `spec.json`, `config.json`, `meta.json`, `generations.jsonl`, `scores.jsonl`, "
      "`finalists.json`, `confirmation.json`.")
    return "\n".join(L)


def write_report(store: RunStore, path: str | None = None) -> str:
    text = build_report(store)
    out = path or str(store.path("report.md"))
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    return out
