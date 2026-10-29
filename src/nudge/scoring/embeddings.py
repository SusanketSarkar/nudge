"""Sentence embedders. The embedder is fixed for a whole experiment and recorded in every score."""
from __future__ import annotations

import re
import zlib
from typing import Protocol, Sequence

import numpy as np

_STOP = frozenset("a an the of to and or in on at is are was were be been it its this that for with as by".split())


class Embedder(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``[n, d]`` float array of L2-normalised embeddings (zero rows for empty text)."""


class HashingEmbedder:
    """Deterministic, offline bag-of-ngrams embedder (feature hashing + sublinear tf).

    A lexical-overlap baseline: it needs no download and is what the tests and the tiny demo use. It is
    **not** a semantic model; for real experiments use :class:`SentenceTransformerEmbedder`.
    """

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.name = f"hashing-v1-d{dim}"

    def _features(self, text: str) -> list[str]:
        words = re.findall(r"[a-z0-9']+", text.lower())
        content = [w for w in words if w not in _STOP]
        feats = ["u:" + w for w in content]
        feats += ["b:" + a + "_" + b for a, b in zip(words, words[1:])]
        joined = " ".join(words)
        feats += ["c:" + joined[i:i + 4] for i in range(max(0, len(joined) - 3))]
        return feats

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            counts: dict[str, int] = {}
            for f in self._features(t):
                counts[f] = counts.get(f, 0) + 1
            for f, c in counts.items():
                h = zlib.crc32(f.encode())
                sign = 1.0 if (h >> 31) & 1 else -1.0
                out[r, h % self.dim] += sign * (1.0 + np.log(c))
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return np.divide(out, norms, out=np.zeros_like(out), where=norms > 0)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # lazy: optional dependency

        self.model = SentenceTransformer(model_name, device=device)
        self.name = f"st:{model_name}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True,
                                            show_progress_bar=False), dtype=np.float32)


class CachedEmbedder:
    """Memoises by exact text so re-scoring and baselines never re-embed."""

    def __init__(self, inner: Embedder) -> None:
        self.inner = inner
        self.name = inner.name
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            for t, v in zip(missing, self.inner.encode(missing)):
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts]) if texts else np.zeros((0, 1), np.float32)


def get_embedder(name: str = "hashing") -> Embedder:
    if name in ("hashing", "hash"):
        return CachedEmbedder(HashingEmbedder())
    return CachedEmbedder(SentenceTransformerEmbedder(name))
