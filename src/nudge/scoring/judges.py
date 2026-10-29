"""LLM-judge evaluator. Never ground truth: fixed prompt, blinded to condition, raw text retained.

``call`` is any ``str -> str`` function, keeping this module free of vendor dependencies.
"""
from __future__ import annotations

import json
import random
import re
from typing import Any, Callable

from nudge.scoring.behavioral import EvalContext
from nudge.utils.hashing import stable_hash

RUBRIC_TEMPLATE = """You are grading a model response. You do NOT know how it was produced.
Criterion: {criterion}

Prompt:
<<<{prompt}>>>

Response:
<<<{output}>>>

Rate how strongly the response exhibits the criterion from 1 (not at all) to 5 (very strongly), and
separately how coherent and on-topic it is from 1 to 5. Reply with ONLY JSON:
{{"criterion_score": <1-5>, "coherence": <1-5>, "explanation": "<one or two sentences>"}}"""

PAIRWISE_TEMPLATE = """Criterion: {criterion}

Prompt:
<<<{prompt}>>>

Response A:
<<<{a}>>>

Response B:
<<<{b}>>>

Which response exhibits the criterion more strongly? Reply with ONLY JSON:
{{"winner": "A" | "B" | "tie", "explanation": "<one sentence>"}}"""


def _parse_json(raw: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError(f"judge returned no JSON: {raw[:120]!r}")
    return json.loads(m.group(0))


class LLMJudge:
    def __init__(self, call: Callable[[str], str], criterion: str, judge_id: str, version: str = "1") -> None:
        self.call, self.criterion = call, criterion
        self.name = f"llm_judge:{judge_id}"
        self.version = f"{version}:{stable_hash(RUBRIC_TEMPLATE + criterion, 6)}"

    def score(self, ctx: EvalContext) -> dict[str, Any]:
        raw = self.call(RUBRIC_TEMPLATE.format(criterion=self.criterion, prompt=ctx.prompt, output=ctx.output))
        try:
            d = _parse_json(raw)
            return {"judge_score": (float(d["criterion_score"]) - 1) / 4,
                    "judge_coherence": (float(d["coherence"]) - 1) / 4,
                    "judge_explanation": d.get("explanation", ""), "judge_raw": raw}
        except Exception as e:  # keep the raw reply so a human can audit parse failures
            return {"judge_error": str(e), "judge_raw": raw}

    def pairwise(self, prompt: str, baseline: str, steered: str, rng: random.Random) -> dict[str, Any]:
        """Randomised A/B order; returns which *condition* won."""
        flip = rng.random() < 0.5
        a, b = (steered, baseline) if flip else (baseline, steered)
        raw = self.call(PAIRWISE_TEMPLATE.format(criterion=self.criterion, prompt=prompt, a=a, b=b))
        d = _parse_json(raw)
        w = d.get("winner", "tie")
        winner = "tie" if w == "tie" else ("steered" if (w == "A") == flip else "baseline")
        return {"winner": winner, "order": "steered_first" if flip else "baseline_first", "raw": raw}


def anthropic_call(model: str, max_tokens: int = 300, api_key: str | None = None) -> Callable[[str], str]:
    """Optional helper (``pip install anthropic``). ``model`` is required; pin it for reproducibility."""
    import anthropic  # lazy

    client = anthropic.Anthropic(api_key=api_key)

    def call(prompt: str) -> str:
        msg = client.messages.create(model=model, max_tokens=max_tokens, temperature=0,
                                     messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

    return call
