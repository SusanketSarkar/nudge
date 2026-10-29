"""Prompt suites with three disjoint partitions: screening / validation / test (held-out)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from nudge.utils.hashing import stable_hash, stable_seed

PARTITIONS = ("screening", "validation", "test")
BUILTIN_DIR = Path(__file__).resolve().parents[3] / "data" / "prompts"


class SplitIntegrityError(ValueError):
    pass


@dataclass(frozen=True)
class Prompt:
    id: str
    text: str
    partition: str
    tags: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)


@dataclass
class PromptSuite:
    name: str
    prompts: list[Prompt]

    def __post_init__(self) -> None:
        self.assert_integrity()

    def partition(self, name: str) -> list[Prompt]:
        if name not in PARTITIONS:
            raise KeyError(f"unknown partition {name!r}")
        return [p for p in self.prompts if p.partition == name]

    def counts(self) -> dict[str, int]:
        return {p: len(self.partition(p)) for p in PARTITIONS}

    @property
    def fingerprint(self) -> str:
        return stable_hash([(p.id, p.text, p.partition) for p in self.prompts], 12)

    def assert_integrity(self) -> None:
        """Ids unique; no prompt text appears in two partitions (case/space-insensitive)."""
        ids = [p.id for p in self.prompts]
        if len(set(ids)) != len(ids):
            raise SplitIntegrityError("duplicate prompt ids")
        seen: dict[str, str] = {}
        for p in self.prompts:
            if p.partition not in PARTITIONS:
                raise SplitIntegrityError(f"{p.id}: unknown partition {p.partition!r}")
            norm = " ".join(p.text.lower().split())
            if norm in seen and seen[norm] != p.partition:
                raise SplitIntegrityError(
                    f"prompt text {p.text!r} appears in both {seen[norm]} and {p.partition}")
            seen[norm] = p.partition

    def limited(self, n_screening: int | None = None, n_validation: int | None = None,
                n_test: int | None = None) -> "PromptSuite":
        caps = {"screening": n_screening, "validation": n_validation, "test": n_test}
        out: list[Prompt] = []
        for part in PARTITIONS:
            items = self.partition(part)
            cap = caps[part]
            out.extend(items[:cap] if cap is not None else items)
        return PromptSuite(self.name, out)


def load_suite(name_or_path: str | Path) -> PromptSuite:
    path = Path(name_or_path)
    if not path.exists():
        path = BUILTIN_DIR / f"{name_or_path}.jsonl"
    if not path.exists():
        avail = sorted(p.stem for p in BUILTIN_DIR.glob("*.jsonl")) if BUILTIN_DIR.exists() else []
        raise FileNotFoundError(f"prompt suite {name_or_path!r} not found (built-in: {avail})")
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        prompts.append(Prompt(r["id"], r["text"], r["partition"], tuple(r.get("tags", ())), r.get("meta", {})))
    return PromptSuite(path.stem, prompts)


def split_flat_list(texts: Iterable[str], name: str = "custom", seed: int = 0,
                    fractions: tuple[float, float, float] = (0.3, 0.2, 0.5)) -> PromptSuite:
    """Deterministically split an unpartitioned list (hash-ordered so it is stable under reordering)."""
    uniq = sorted({" ".join(t.split()) for t in texts}, key=lambda t: stable_seed(seed, t))
    n = len(uniq)
    a, b = round(n * fractions[0]), round(n * (fractions[0] + fractions[1]))
    prompts = []
    for i, t in enumerate(uniq):
        part = PARTITIONS[0] if i < a else PARTITIONS[1] if i < b else PARTITIONS[2]
        prompts.append(Prompt(f"{name}-{stable_hash(t, 6)}", t, part))
    return PromptSuite(name, prompts)
