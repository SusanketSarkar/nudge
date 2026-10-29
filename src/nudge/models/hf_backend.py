"""Hugging Face / PyTorch steering backend.

Intervention (``hook_point = "mlp_out"``)
-----------------------------------------
For a gated-MLP decoder block ``y = W_down @ (act(W_gate x) * (W_up x))`` let ``h`` be the vector fed
to ``down_proj`` (the *neuron activations*) and ``W_down`` the ``[hidden, intermediate]`` weight of the
``nn.Linear``. Pushing neuron ``i`` by ``alpha`` means

    y' = y + alpha * W_down[:, i]        (equivalently: h_i -> h_i + alpha)

applied at every position, prompt and generated. This is a change to the residual-stream contribution
at inference time, not a weight update.

Dose units (declared, never implicit)
--------------------------------------
``raw_activation``
    ``alpha = dose``. Adds ``dose`` to the neuron's activation. Scale depends on the neuron.
``rel_mlp_out_norm``
    ``delta = dose * R_layer * W_down[:, i] / ||W_down[:, i]||`` where ``R_layer`` is the *median
    per-token L2 norm of that layer's MLP output* measured on calibration texts. Every neuron in a
    layer then gets the same push magnitude, which makes candidates and random controls comparable.

Random-direction control
------------------------
Same magnitude as the neuron intervention it shadows, but along a seeded random unit vector in the
residual (MLP-output) space. See ``docs/METHODOLOGY.md``.

Validity guards
---------------
Adding at ``down_proj`` output equals "add to a neuron" only if nothing non-linear sits between
``down_proj`` and the residual add. We therefore refuse blocks with a post-FFN norm or layer scale
(Gemma-2/3 style) unless ``allow_post_ffn_norm=True``, and refuse quantised ``down_proj`` layers.
"""
from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np
import torch

from nudge.models.base import GenerationConfig, GenerationResult, SteeringBackend
from nudge.steering.hooks import additive_mlp_out
from nudge.steering.interventions import KIND_RANDOM, Intervention, as_tuple
from nudge.utils.hashing import stable_hash, stable_seed
from nudge.utils.system import available_memory_bytes, estimate_param_count, fmt_bytes

DOSE_UNITS = {
    "raw_activation": "alpha added to the neuron's activation h_i (delta = alpha * W_down[:, i]).",
    "rel_mlp_out_norm": (
        "delta = dose * median_token_L2_norm(layer MLP output) * unit(W_down[:, i]); "
        "the same push magnitude for every neuron in a layer."
    ),
}

DEFAULT_CALIBRATION_TEXTS = (
    "The history of the printing press changed how ideas moved between people.",
    "To make a good loaf of bread you need flour, water, salt and patience.",
    "She opened the window and listened to the rain falling on the quiet street.",
    "In 2019 the committee published a report on regional transport planning.",
    "def add(a, b):\n    return a + b\n",
    "Photosynthesis converts light energy into chemical energy stored in sugars.",
    "He wasn't sure whether the train would arrive on time, but he waited anyway.",
    "What is the best way to learn a new language as an adult?",
)


def _resolve(obj: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


class HFBackend(SteeringBackend):
    name = "hf"

    def __init__(self, model: torch.nn.Module, tokenizer: Any, *, model_id: str | None = None,
                 layers_path: str = "model.layers", down_proj_path: str = "mlp.down_proj",
                 calibration_texts: Sequence[str] = DEFAULT_CALIBRATION_TEXTS,
                 allow_post_ffn_norm: bool = False, min_layer: int = 0) -> None:
        self.model = model.eval()
        self.tok = tokenizer
        self._model_id = model_id or getattr(getattr(model, "config", None), "_name_or_path", "") or "unknown"
        self.calibration_texts = tuple(calibration_texts)
        self.min_layer = min_layer
        layers = _resolve(model, layers_path)
        self._down: dict[int, torch.nn.Module] = {}
        for i, layer in enumerate(layers):
            down = _resolve(layer, down_proj_path)
            if not isinstance(down, torch.nn.Linear) or not torch.is_floating_point(down.weight):
                raise TypeError(
                    f"layer {i}: {down_proj_path} is {type(down).__name__}; Nudge needs a plain "
                    "floating-point nn.Linear (quantised down_proj layers are not supported)."
                )
            if down.bias is not None:
                raise ValueError(f"layer {i}: down_proj has a bias; additive neuron steering would be ambiguous.")
            if not allow_post_ffn_norm:
                for attr in ("post_feedforward_layernorm", "post_ffn_layernorm", "ffn_layer_scale"):
                    if hasattr(layer, attr):
                        raise ValueError(
                            f"layer {i} has `{attr}` after the MLP; additive steering at down_proj output "
                            "is no longer equivalent to steering a neuron. Pass allow_post_ffn_norm=True "
                            "to proceed knowingly."
                        )
            self._down[i] = down
        if not self._down:
            raise ValueError(f"no layers found at {layers_path!r}")
        self.hidden_size = self._down[0].out_features
        self.intermediate_size = self._down[0].in_features
        self._ref_norms: dict[int, float] = {}

    # -- construction -----------------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, model_id: str, *, device: str = "auto", dtype: str = "auto",
                        force: bool = False, **kwargs: Any) -> "HFBackend":
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        cfg = AutoConfig.from_pretrained(model_id)
        if device == "auto":
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                      else "cpu")
        torch_dtype = {"auto": None, "float32": torch.float32, "float16": torch.float16,
                       "bfloat16": torch.bfloat16}[dtype]
        if torch_dtype is None:
            torch_dtype = torch.float32 if device == "cpu" else torch.bfloat16
        params = estimate_param_count(cfg)
        avail = available_memory_bytes()
        if params and avail and device == "cpu":
            need = params * torch.empty((), dtype=torch_dtype).element_size()
            if need > 0.85 * avail and not force:
                raise MemoryError(
                    f"{model_id} needs ~{fmt_bytes(need)} but only {fmt_bytes(avail)} is available. "
                    "Use a smaller model/dtype or pass force=True."
                )
        tok = AutoTokenizer.from_pretrained(model_id)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)
        except TypeError:  # older transformers
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
        model.to(device)
        return cls(model, tok, model_id=model_id, **kwargs)

    # -- SteeringBackend --------------------------------------------------------------------------
    def list_supported_layers(self) -> list[int]:
        return [i for i in sorted(self._down) if i >= self.min_layer]

    def list_neurons(self, layer_id: int) -> range:
        return range(self._down[layer_id].in_features)

    def supported_dose_units(self) -> dict[str, str]:
        return dict(DOSE_UNITS)

    def get_model_metadata(self) -> dict[str, Any]:
        down0 = self._down[0]
        cfg = getattr(self.model, "config", None)
        return {
            "backend": self.name,
            "model_id": self._model_id,
            "model_class": type(self.model).__name__,
            "tokenizer_id": getattr(self.tok, "name_or_path", type(self.tok).__name__),
            "n_layers": len(self._down),
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "dtype": str(down0.weight.dtype).replace("torch.", ""),
            "device": str(down0.weight.device),
            "hook_point": "mlp_out (additive, output of down_proj)",
            "dose_units": self.supported_dose_units(),
            "calibration_texts_hash": stable_hash(list(self.calibration_texts), 10),
            "weights_fingerprint": self._fingerprint(),
            "transformers_config_hash": stable_hash(cfg.to_dict(), 10) if hasattr(cfg, "to_dict") else None,
        }

    def _fingerprint(self) -> str:
        w = self._down[0].weight.detach().float().cpu().numpy()
        return stable_hash(np.round(w[:8, :16], 6).tolist(), 12)

    # -- steering vectors -------------------------------------------------------------------------
    def mlp_out_norm(self, layer: int) -> float:
        """Median per-token L2 norm of the layer's MLP output on the calibration texts (cached)."""
        if layer not in self._ref_norms:
            norms: list[torch.Tensor] = []
            handle = self._down[layer].register_forward_hook(
                lambda _m, _a, out: norms.append(out.detach().float().norm(dim=-1).flatten().cpu()))
            try:
                with torch.inference_mode():
                    for text in self.calibration_texts:
                        enc = self._encode(text)
                        self.model(**enc)
            finally:
                handle.remove()
            self._ref_norms[layer] = float(torch.cat(norms).median())
        return self._ref_norms[layer]

    def delta_vector(self, iv: Intervention) -> torch.Tensor:
        """The vector added to the MLP output at ``iv.layer`` (float32, shape ``[hidden]``)."""
        if iv.layer not in self._down:
            raise ValueError(f"layer {iv.layer} not in model (0..{len(self._down) - 1})")
        if iv.dose_unit not in DOSE_UNITS:
            raise ValueError(f"unsupported dose_unit {iv.dose_unit!r}; choose from {sorted(DOSE_UNITS)}")
        w = self._down[iv.layer].weight
        if not 0 <= iv.neuron < w.shape[1]:
            raise ValueError(f"neuron {iv.neuron} out of range for layer {iv.layer} ({w.shape[1]})")
        col = w[:, iv.neuron].detach().float()
        col_norm = col.norm().clamp_min(1e-12)
        if iv.dose_unit == "raw_activation":
            magnitude = iv.dose * col_norm  # signed
        else:
            magnitude = torch.as_tensor(iv.dose * self.mlp_out_norm(iv.layer))
        if iv.kind == KIND_RANDOM:
            g = torch.Generator(device="cpu").manual_seed(stable_seed(iv.layer, iv.neuron, iv.random_seed))
            direction = torch.randn(col.shape[0], generator=g)
            direction = (direction / direction.norm()).to(col.device)
        else:
            direction = col / col_norm
        return (magnitude * direction).to(col.device)

    def _deltas(self, ivs: Sequence[Intervention]) -> dict[int, torch.Tensor]:
        out: dict[int, torch.Tensor] = {}
        for iv in ivs:
            d = self.delta_vector(iv)
            out[iv.layer] = out[iv.layer] + d if iv.layer in out else d
        return out

    # -- generation -------------------------------------------------------------------------------
    def _render(self, prompt: str, cfg: GenerationConfig) -> str:
        if cfg.use_chat_template and getattr(self.tok, "chat_template", None):
            msgs = ([{"role": "system", "content": cfg.system_prompt}] if cfg.system_prompt else [])
            msgs.append({"role": "user", "content": prompt})
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return prompt

    def _encode(self, text: str, max_len: int = 128) -> dict[str, torch.Tensor]:
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=max_len)
        dev = self._down[0].weight.device
        return {k: v.to(dev) for k, v in enc.items() if k in ("input_ids", "attention_mask")}

    def _generate(self, prompt: str, ivs: Sequence[Intervention], cfg: GenerationConfig) -> GenerationResult:
        t0 = time.perf_counter()
        try:
            enc = self._encode(self._render(prompt, cfg), max_len=2048)
            n_prompt = int(enc["input_ids"].shape[1])
            deltas = self._deltas(ivs)
            torch.manual_seed(cfg.seed)
            kw: dict[str, Any] = {"max_new_tokens": cfg.max_new_tokens}
            if cfg.temperature > 0:
                kw.update(do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p)
            else:
                kw.update(do_sample=False)
            pad = getattr(self.tok, "pad_token_id", None)
            if pad is None:
                pad = getattr(self.tok, "eos_token_id", None)
            if pad is not None:
                kw["pad_token_id"] = pad
            with torch.inference_mode(), additive_mlp_out(self._down, deltas):
                out = self.model.generate(**enc, **kw)
            ids = out[0, n_prompt:].tolist()
            text = self.tok.decode(ids, skip_special_tokens=True)
            return GenerationResult(text=text, token_ids=ids, n_prompt_tokens=n_prompt,
                                    latency_s=time.perf_counter() - t0)
        except Exception as e:  # recorded, never swallowed silently
            return GenerationResult(text="", status="error", error=f"{type(e).__name__}: {e}",
                                    latency_s=time.perf_counter() - t0)

    def run_baseline(self, prompt: str, generation_config: GenerationConfig) -> GenerationResult:
        return self._generate(prompt, (), generation_config)

    def run_steered(self, prompt: str, intervention: "Intervention | Sequence[Intervention]",
                    generation_config: GenerationConfig) -> GenerationResult:
        return self._generate(prompt, as_tuple(intervention), generation_config)

    def logits(self, prompt: str, interventions: "Intervention | Sequence[Intervention] | None" = None,
               ) -> torch.Tensor:
        """Last-position next-token logits (float32, CPU); used by the self-checks."""
        deltas = self._deltas(as_tuple(interventions))
        with torch.inference_mode(), additive_mlp_out(self._down, deltas):
            out = self.model(**self._encode(prompt))
        return out.logits[0, -1].float().cpu()
