"""Scores stored generations; scoring is decoupled from generation so it can be redone any time."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from nudge.experiments.storage import RunStore, ScoreRecord
from nudge.scoring.behavioral import EvalContext, Evaluator, coherence_features
from nudge.scoring.embeddings import Embedder
from nudge.scoring.semantic import FORMULA_VERSION, SemanticScorer
from nudge.targets.schema import TargetSpec
from nudge.utils.hashing import stable_hash


class ScoringPipeline:
    def __init__(self, spec: TargetSpec, embedder: Embedder, evaluators: Sequence[Evaluator] = ()) -> None:
        self.spec, self.embedder, self.evaluators = spec, embedder, tuple(evaluators)
        self.semantic = SemanticScorer(embedder, spec)
        self.scorer_id = stable_hash({
            "spec": spec.spec_hash, "embedder": embedder.name, "formula": FORMULA_VERSION,
            "coherence": "heuristic-v1", "evaluators": [(e.name, e.version) for e in self.evaluators],
        }, 12)

    def score_store(self, store: RunStore, batch_size: int = 64) -> int:
        """Score every OK generation lacking a score under this scorer_id. Returns #new scores."""
        todo = [g for g in store.generations()
                if g.status == "ok" and not store.has_score(g.key, self.scorer_id)]
        n = 0
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            sem = self.semantic.score([g.output for g in batch])
            rel = np.sum(self.embedder.encode([g.output for g in batch]) *
                         self.embedder.encode([g.prompt for g in batch]), axis=1)
            recs = []
            for g, s, r in zip(batch, sem, rel):
                coh = coherence_features(g.output)
                behavior, raw = {}, {}
                ctx = EvalContext(g.prompt, g.output, g.prompt_meta, self.spec)
                for ev in self.evaluators:
                    out = ev.score(ctx)
                    behavior.update({k: v for k, v in out.items()
                                     if isinstance(v, (int, float, bool)) and not k.startswith("judge_raw")})
                    if out:
                        raw[ev.name] = out
                recs.append(ScoreRecord(
                    key=g.key, scorer_id=self.scorer_id, s_target=s.s_target, s_contrast=s.s_contrast, j=s.j,
                    target_sims=s.target_sims, contrast_sims=s.contrast_sims, coherence=coh["coherence"],
                    relevance=float(r), flags=coh["flags"], behavior=behavior, evaluator_raw=raw))
            store.add_scores(recs)
            n += len(recs)
        return n
