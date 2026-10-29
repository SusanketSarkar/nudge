"""Join generations, scores and baselines; build candidate tables and the leaderboard."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nudge.experiments.storage import COND_BASELINE, COND_CONTROL, COND_STEERED, RunStore
from nudge.scoring.controls import benjamini_hochberg, control_comparison
from nudge.utils.hashing import stable_hash


@dataclass
class RankingParams:
    min_valid_prompts: int = 5          # minimum number of valid outputs per (neuron, dose)
    min_coherence: float = 0.0          # candidate's mean heuristic coherence must reach this
    control_percentile: float = 0.95    # "beats controls" threshold
    min_control_seeds: int = 3
    min_common_prompts: int = 2


ROW_COLS = ["key", "stage", "partition", "prompt_id", "condition", "n_iv", "layer", "neuron", "dose",
            "dose_unit", "control_seed", "iv_id", "ok", "j", "s_target", "s_contrast", "coherence",
            "relevance", "answerable", "behavior_success", "flags", "output", "baseline_key", "dj", "d_coherence",
            "d_relevance", "d_behavior"]


def build_rows(store: RunStore, scorer_id: str) -> pd.DataFrame:
    scores = store.scores(scorer_id)
    recs = []
    for g in store.generations():
        s = scores.get(g.key)
        ivs = g.interventions
        recs.append({
            "key": g.key, "stage": g.stage, "partition": g.partition, "prompt_id": g.prompt_id,
            "condition": g.condition, "n_iv": len(ivs), "layer": g.layer, "neuron": g.neuron, "dose": g.dose,
            "dose_unit": ivs[0]["dose_unit"] if len(ivs) == 1 else None, "control_seed": g.control_seed,
            "iv_id": "+".join(f"L{i['layer']}.N{i['neuron']}@{i['dose']:+g}"
                              + (f"~rand{i['random_seed']}" if i.get("random_seed") is not None else "")
                              for i in ivs) or "baseline",
            "ok": g.status == "ok" and s is not None,
            "j": s.j if s else np.nan, "s_target": s.s_target if s else np.nan,
            "s_contrast": s.s_contrast if s else np.nan, "coherence": s.coherence if s else np.nan,
            "relevance": s.relevance if s else np.nan,
            "answerable": g.prompt_meta.get("answerable"),
            "behavior_success": (s.behavior.get("behavior_success") if s else None),
            "flags": ",".join(s.flags) if s else "error", "output": g.output,
            "_cfg": stable_hash(g.gen_config, 10),
        })
    df = pd.DataFrame(recs)
    if df.empty:
        return pd.DataFrame(columns=ROW_COLS)
    df["behavior_success"] = pd.to_numeric(df["behavior_success"], errors="coerce")
    base = df[(df.condition == COND_BASELINE) & df.ok].set_index(["prompt_id", "_cfg"])
    base = base[~base.index.duplicated(keep="last")]
    idx = pd.MultiIndex.from_frame(df[["prompt_id", "_cfg"]])
    for col, new in (("j", "dj"), ("coherence", "d_coherence"), ("relevance", "d_relevance"),
                     ("behavior_success", "d_behavior")):
        b = base[col].reindex(idx).to_numpy()
        df[new] = np.where(df.condition == COND_BASELINE, np.nan, df[col].to_numpy() - b)
    df["baseline_key"] = base["key"].reindex(idx).to_numpy()
    df.loc[~df.ok, ["dj", "d_coherence", "d_relevance", "d_behavior"]] = np.nan
    return df.drop(columns=["_cfg"])


def _bmean(g: pd.DataFrame, answerable: bool) -> float:
    sub = g[(g.answerable == answerable) & g.d_behavior.notna()]
    return float(sub.d_behavior.mean()) if len(sub) else np.nan


def candidate_table(rows: pd.DataFrame, params: RankingParams, partition: str = "screening") -> pd.DataFrame:
    """One row per (layer, neuron, dose), single-neuron interventions only."""
    cand = rows[(rows.partition == partition) & (rows.condition == COND_STEERED) & (rows.n_iv == 1)]
    ctrl = rows[(rows.partition == partition) & (rows.condition == COND_CONTROL) & rows.ok & rows.dj.notna()]
    pivot = (ctrl.pivot_table(index=["layer", "dose", "control_seed"], columns="prompt_id", values="dj")
             if len(ctrl) else None)
    out = []
    for (layer, neuron, dose), g in cand.groupby(["layer", "neuron", "dose"]):
        allok = g[g.ok & g.dj.notna()]
        valid = allok[allok.coherence >= params.min_coherence] if params.min_coherence > 0 else allok
        row = {
            "layer": int(layer), "neuron": int(neuron), "dose": float(dose),
            "neuron_id": f"L{int(layer)}.N{int(neuron)}", "n_total": len(g), "n_prompts": len(valid),
            "mean_dj": float(valid.dj.mean()) if len(valid) else np.nan,
            "std_dj": float(valid.dj.std(ddof=1)) if len(valid) > 1 else np.nan,
            "success_rate": float((valid.dj > 0).mean()) if len(valid) else np.nan,
            "mean_coherence": float(allok.coherence.mean()) if len(allok) else np.nan,
            "d_coherence": float(allok.d_coherence.mean()) if len(allok) else np.nan,
            "d_relevance": float(allok.d_relevance.mean()) if len(allok) else np.nan,
            "behavior_delta": float(allok.d_behavior.mean()) if allok.d_behavior.notna().any() else np.nan,
            "behavior_delta_answerable": _bmean(allok, True),
            "behavior_delta_unanswerable": _bmean(allok, False),
            "control_percentile": np.nan, "p_control": np.nan, "n_controls": 0,
        }
        row["sem_dj"] = row["std_dj"] / np.sqrt(row["n_prompts"]) if row["n_prompts"] > 1 else np.nan
        if pivot is not None and (layer, dose) in pivot.index.droplevel(2):
            sub = pivot.loc[(layer, dose)]
            common = [p for p in allok.prompt_id if p in sub.columns]
            if len(common) >= params.min_common_prompts:
                cm = sub[common].dropna().mean(axis=1)
                if len(cm) >= params.min_control_seeds:
                    cand_mean = float(allok.set_index("prompt_id").loc[common, "dj"].mean())
                    cc = control_comparison(cand_mean, cm.to_numpy())
                    row.update(control_percentile=cc["percentile"], p_control=cc["p_value"],
                               n_controls=int(cc["n_controls"]))
        row["eligible"] = bool(row["n_prompts"] >= params.min_valid_prompts
                               and (row["mean_coherence"] >= params.min_coherence))
        out.append(row)
    df = pd.DataFrame(out)
    if df.empty:
        return df
    df["q_value"] = benjamini_hochberg(df["p_control"].to_numpy())
    return df


def leaderboard(table: pd.DataFrame, params: RankingParams,
                validation_status: dict[str, str] | None = None) -> pd.DataFrame:
    """Best sign/dose per neuron among eligible candidates, ranked by mean ΔJ (all metrics shown)."""
    if table.empty:
        return table
    t = table[table.eligible & table.mean_dj.notna()]
    if t.empty:
        return t
    best = t.loc[t.groupby(["layer", "neuron"]).mean_dj.idxmax()].copy()
    best["best_sign"] = np.where(best.dose >= 0, "+", "-")
    best["beats_control"] = np.where(best.control_percentile.isna(), None,
                                     best.control_percentile >= params.control_percentile)
    best = best.sort_values("mean_dj", ascending=False).reset_index(drop=True)
    best.insert(0, "rank", np.arange(1, len(best) + 1))
    vs = validation_status or {}
    best["validation_status"] = [vs.get(f"{r.neuron_id}@{r.dose:+g}", "unvalidated") for r in best.itertuples()]
    return best


def profile_matrix(rows: pd.DataFrame, candidates: pd.DataFrame, partition: str = "screening") -> np.ndarray:
    """Per-prompt ΔJ vector for each candidate row (NaN -> 0): its 'behavioural profile'."""
    cand = rows[(rows.partition == partition) & (rows.condition == COND_STEERED) & (rows.n_iv == 1) & rows.ok]
    prompts = sorted(cand.prompt_id.unique())
    piv = cand.pivot_table(index=["layer", "neuron", "dose"], columns="prompt_id", values="dj").reindex(columns=prompts)
    mats = []
    for r in candidates.itertuples():
        try:
            mats.append(piv.loc[(r.layer, r.neuron, r.dose)].fillna(0.0).to_numpy())
        except KeyError:
            mats.append(np.zeros(len(prompts)))
    return np.vstack(mats) if mats else np.zeros((0, len(prompts)))
