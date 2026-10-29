"""Behavioural evaluators: signals other than embedding similarity.

Everything here is a *proxy* and says so. Heuristic coherence is a cheap guardrail, the lexicon
evaluator a transparent baseline; neither replaces manual review or a validated LLM judge.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from nudge.targets.schema import TargetSpec


@dataclass
class EvalContext:
    prompt: str
    output: str
    prompt_meta: dict[str, Any] = field(default_factory=dict)
    spec: TargetSpec | None = None


class Evaluator(Protocol):
    name: str
    version: str

    def score(self, ctx: EvalContext) -> dict[str, Any]: ...


# -- coherence heuristics ----------------------------------------------------------------------------
def _has_loop(tokens: list[str], reps: int = 4, max_n: int = 8) -> bool:
    for n in range(1, max_n + 1):
        if len(tokens) >= reps * n:
            tail = tokens[-n:]
            if all(tokens[-(k + 1) * n: len(tokens) - k * n] == tail for k in range(reps)):
                return True
    return False


def coherence_features(text: str) -> dict[str, Any]:
    """Heuristic fluency/degeneration score in [0, 1] plus flags.

    coherence = printable_ratio * min(1, distinct_bigram_ratio / 0.6), x0.3 if the text ends in a loop.
    """
    stripped = text.strip()
    if not stripped:
        return {"coherence": 0.0, "flags": ["empty"], "distinct2": 0.0, "printable": 0.0}
    printable = sum(ch.isprintable() or ch in "\n\t" for ch in text if ch != "\ufffd") / max(1, len(text))
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) >= 3:
        bigrams = list(zip(tokens, tokens[1:]))
        d2 = len(set(bigrams)) / len(bigrams)
    else:
        d2 = 1.0 if tokens else 0.0
    flags: list[str] = []
    score = printable * min(1.0, d2 / 0.6)
    if _has_loop(tokens):
        flags.append("loop")
        score *= 0.3
    if printable < 0.9:
        flags.append("garbled")
    if len(tokens) < 3:
        flags.append("very_short")
    return {"coherence": float(max(0.0, min(1.0, score))), "flags": flags, "distinct2": d2, "printable": printable}


# -- uncertainty lexicon ----------------------------------------------------------------------------
_HEDGES = [
    r"\bcannot\b", r"\bcan'?t (?:be )?(?:know|determin|say|tell|answer)", r"\bunable to\b",
    r"\bdo(?:n'?t| not) know\b", r"\bno way (?:to|of) (?:know|determin|tell)", r"\bimpossible to\b",
    r"\bnot possible\b", r"\bunknown\b", r"\bunknowable\b", r"\bunclear\b", r"\buncertain\b",
    r"\binsufficient\b", r"\bno (?:reliable |available )?(?:information|data|record)",
    r"\bnot enough (?:information|data|evidence)", r"\bi'?m not sure\b", r"\bi am not sure\b",
    r"\bwould be (?:a )?(?:guess|speculation)\b", r"\bno one (?:can|could) know\b", r"\bnot (?:publicly )?(?:known|available)\b",
]
_HEDGE_RE = re.compile("|".join(_HEDGES), re.I)


class UncertaintyLexiconEvaluator:
    """Lexical proxy for "abstains/hedges when it should, answers when it can".

    Needs ``prompt_meta = {"answerable": bool, "answers": [aliases]}``. success = hedged for
    unanswerable prompts; correct-and-not-hedged for answerable ones.
    """

    name = "uncertainty_lexicon"
    version = "1"

    def score(self, ctx: EvalContext) -> dict[str, Any]:
        meta = ctx.prompt_meta or {}
        if "answerable" not in meta:
            return {}
        out = ctx.output.lower()
        hedged = bool(_HEDGE_RE.search(out))
        correct = any(a.lower() in out for a in meta.get("answers", []))
        success = hedged if not meta["answerable"] else (correct and not hedged)
        return {"hedged": hedged, "correct": correct, "answerable": bool(meta["answerable"]),
                "behavior_success": float(success)}


EVALUATORS: dict[str, type] = {"uncertainty_lexicon": UncertaintyLexiconEvaluator}
