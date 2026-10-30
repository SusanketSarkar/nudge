"""Turn a natural-language goal into a :class:`TargetSpec`.

* Built-in **presets** are hand-written and are the recommended starting point.
* ``propose_spec(goal, llm=...)`` asks any ``str -> str`` callable (e.g. an LLM client) to *propose*
  descriptions. The result is flagged ``reviewed=False`` and ``origin="llm_proposed"``: a human is meant
  to inspect and edit it (UI page 1) before it is used. Nothing is silently trusted.
* Without an LLM we return an explicit skeleton that lint will flag as needing work.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable

from nudge.targets.schema import TargetSpec

PRESETS: dict[str, TargetSpec] = {
    "secret_society_thriller": TargetSpec(
        target_name="secret_society_thriller",
        goal="Make fictional prose sound like a secret-society conspiracy thriller.",
        target_descriptions=[
            "A tense conspiracy thriller about a secret society, hidden symbols and whispered warnings.",
            "Shadowy figures in a hidden order meet in secret, and nobody can be trusted.",
            "The narrator realises they are being watched by an ancient secret brotherhood.",
        ],
        contrast_descriptions=[
            "Ordinary everyday description of a mundane scene in plain, calm language.",
            "A cheerful, straightforward account of a normal day with nothing unusual in it.",
            "Neutral factual prose describing ordinary places and routine activities.",
        ],
        task_type="stylistic",
        guardrails=["Do not reward incoherent or irrelevant output."],
        evaluation_dimensions=["target_semantic_shift", "coherence", "relevance"],
        reviewed=True, origin="preset"),
    "uncertainty_when_insufficient_evidence": TargetSpec(
        target_name="uncertainty_when_insufficient_evidence",
        goal=("Increase the tendency to express uncertainty when the available evidence is insufficient, "
              "without reducing accuracy on answerable questions."),
        target_descriptions=[
            "The answer explicitly acknowledges that the available evidence is insufficient to answer.",
            "I cannot know this; the response distinguishes known facts from uncertain claims.",
            "It is not possible to determine this from the information given, so I will not guess.",
        ],
        contrast_descriptions=[
            "The response confidently asserts a specific unsupported answer as if it were established fact.",
            "A definite, authoritative answer is stated with no hedging, presenting speculation as fact.",
            "The answer is simply stated as certain, even though nothing supports it.",
        ],
        task_type="behavioral",
        guardrails=["Do not increase unnecessary abstention on answerable questions.",
                    "Do not reward incoherent or irrelevant output."],
        evaluation_dimensions=["target_semantic_shift", "behavioral_success", "coherence", "answerability_tradeoff"],
        behavior_evaluator="uncertainty_lexicon", reviewed=True, origin="preset"),
    "terse": TargetSpec(
        target_name="terse",
        goal="Make responses more terse.",
        target_descriptions=["A very short, blunt answer in a few words.",
                             "Minimal wording, no elaboration, straight to the point."],
        contrast_descriptions=["A long, elaborate, wordy explanation with many details and digressions.",
                               "A flowery, verbose response that takes a long time to get to the point."],
        task_type="stylistic", reviewed=True, origin="preset"),
}


def list_presets() -> list[str]:
    return sorted(PRESETS)


def get_preset(name: str) -> TargetSpec:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; available: {list_presets()}")
    return PRESETS[name].model_copy(deep=True)


_PROMPT = """You help design a controlled experiment on a language model.
Goal: {goal}

Return ONLY JSON with keys:
  "target_descriptions": 3 diverse sentences describing text that exhibits the goal,
  "contrast_descriptions": 3 diverse sentences describing text that does NOT (the opposite or the neutral baseline),
  "guardrails": 1-3 things that must not get worse,
  "task_type": "behavioral" or "stylistic".
Keep descriptions about the *content/behavior of a response*, not about neurons or models."""


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:48] or "target"


def _endpoint(base_url: str, suffix: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith(suffix) else base + suffix


def list_openai_models(base_url: str, api_key: str | None = None, timeout: float = 30.0) -> list[str]:
    """Model ids exposed by an OpenAI-compatible server (vLLM, llama.cpp, ...)."""
    import urllib.error
    import urllib.request

    url = _endpoint(base_url, "/models")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"model discovery failed ({e.code}) at {url}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"could not reach LLM at {url}: {e.reason}") from e
    return [str(m["id"]) for m in data.get("data", []) if m.get("id")]


def openai_compatible_call(base_url: str, model: str, api_key: str | None = None,
                           timeout: float = 120.0, max_tokens: int = 600,
                           temperature: float = 0.0) -> Callable[[str], str]:
    """Return a ``str -> str`` callable backed by an OpenAI-compatible chat endpoint.

    Works with a local vLLM server (``--base-url http://localhost:8000/v1``), llama.cpp, or a hosted
    OpenAI-style API. No extra dependency is needed: it uses ``urllib`` from the standard library.
    """
    import urllib.error
    import urllib.request

    url = _endpoint(base_url, "/chat/completions")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def call(prompt: str) -> str:
        body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                           "temperature": temperature, "max_tokens": max_tokens}).encode()
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            raise RuntimeError(f"LLM request failed ({e.code}): {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"could not reach LLM at {url}: {e.reason}") from e
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"unexpected LLM response: {str(data)[:300]}") from e

    return call


def propose_spec(goal: str, llm: Callable[[str], str] | None = None) -> TargetSpec:
    if llm is None:
        return TargetSpec(
            target_name=_slug(goal), goal=goal,
            target_descriptions=[f"A response that clearly shows this property: {goal}"],
            contrast_descriptions=["An ordinary neutral response without that property."],
            reviewed=False, origin="skeleton")
    raw = llm(_PROMPT.format(goal=goal))
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError("LLM did not return JSON")
    d = json.loads(m.group(0))
    return TargetSpec(
        target_name=_slug(goal), goal=goal,
        target_descriptions=d["target_descriptions"], contrast_descriptions=d["contrast_descriptions"],
        guardrails=d.get("guardrails", []),
        task_type=d.get("task_type", "stylistic") if d.get("task_type") in ("behavioral", "stylistic") else "stylistic",
        reviewed=False, origin="llm_proposed")
