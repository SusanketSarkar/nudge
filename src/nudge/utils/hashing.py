"""Small deterministic hashing / seeding helpers (never use Python's salted ``hash``)."""
from __future__ import annotations

import hashlib
import json
import zlib
from typing import Any


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def stable_hash(obj: Any, n: int = 16) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:n]


def stable_seed(*parts: Any) -> int:
    """Deterministic 32-bit seed from arbitrary parts."""
    return zlib.crc32(canonical_json(list(parts)).encode("utf-8")) & 0xFFFFFFFF
