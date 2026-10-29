"""A tiny randomly-initialised Llama + byte tokenizer.

This exists so the *entire pipeline* (hooks, runner, storage, scoring, UI) can be exercised and tested
on any machine in seconds with no downloads. It is a plumbing fixture: its text is gibberish and any
"effect" found on it is meaningless scientifically. Use :meth:`HFBackend.from_pretrained` for real work.
"""
from __future__ import annotations

import torch
from transformers import BatchEncoding, LlamaConfig, LlamaForCausalLM

from nudge.models.hf_backend import HFBackend

PAD, BOS, EOS = 0, 1, 2
_OFFSET = 3


class ByteTokenizer:
    """Minimal byte-level tokenizer implementing the slice of the HF API Nudge uses."""

    name_or_path = "nudge-byte-tokenizer"
    chat_template = None
    pad_token_id = PAD
    eos_token_id = EOS
    vocab_size = 256 + _OFFSET

    def __call__(self, text: str, return_tensors: str | None = None, truncation: bool = False,
                 max_length: int | None = None, **_: object) -> BatchEncoding:
        ids = [BOS] + [b + _OFFSET for b in text.encode("utf-8")]
        if truncation and max_length:
            ids = ids[-max_length:]
        t = torch.tensor([ids])
        return BatchEncoding({"input_ids": t, "attention_mask": torch.ones_like(t)})

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        data = bytes(i - _OFFSET for i in ids if i >= _OFFSET)
        return data.decode("utf-8", errors="replace")


def build_tiny_backend(seed: int = 0, n_layers: int = 6, hidden: int = 64, intermediate: int = 192,
                       **kwargs) -> HFBackend:
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=ByteTokenizer.vocab_size, hidden_size=hidden,
                      intermediate_size=intermediate, num_hidden_layers=n_layers,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=1024,
                      pad_token_id=PAD, bos_token_id=BOS, eos_token_id=EOS, tie_word_embeddings=False)
    model = LlamaForCausalLM(cfg).eval()
    return HFBackend(model, ByteTokenizer(), model_id=f"nudge-tiny-random-llama-seed{seed}", **kwargs)
