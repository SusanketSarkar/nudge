"""Memory / device checks so we fail early instead of freezing a laptop."""
from __future__ import annotations

import os


def available_memory_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


def estimate_param_count(config) -> int | None:
    """Rough decoder-only parameter estimate from a HF config (None if fields are missing)."""
    try:
        d = config.hidden_size
        n = config.num_hidden_layers
        ff = config.intermediate_size
        v = config.vocab_size
        kv = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head = d // config.num_attention_heads
        attn = d * d * 2 + 2 * d * kv * head
        mlp = 3 * d * ff
        emb = v * d * (1 if getattr(config, "tie_word_embeddings", False) else 2)
        return int(n * (attn + mlp) + emb)
    except Exception:
        return None


def fmt_bytes(n: float | None) -> str:
    if n is None:
        return "unknown"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PiB"
