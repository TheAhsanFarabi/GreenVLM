"""
GreenVLM: HuggingFace backend for models not served by Ollama (e.g. InternVL).

Exposes `hf_query(...)`, which returns a response dict shaped exactly like the
Ollama `/api/chat` reply used by `main_benchmark.py`:

    {"response": str, "prompt_eval_duration": ns, "eval_duration": ns, "eval_count": int}

The prefill ("Seeing") vs decode ("Speaking") split is measured by time-to-first-token:
a streamer runs `generate` in a worker thread while the main thread records when the
first token arrives (prefill) and when generation ends (decode) — the direct analogue
of Ollama's `prompt_eval_duration` / `eval_duration`. Wrap an energy tracker around the
call (as main_benchmark already does) to attribute measured Joules to each phase.

Dynamic resolution: the image is passed to the processor at whatever size the caller
resized it to, so InternVL's native tiling scales visual tokens with resolution.
"""

import io
import time
import base64
import threading
from typing import Dict, Any, Optional

from PIL import Image

_MODEL = None
_PROC = None
_MODEL_ID: Optional[str] = None


def load_model(model_id: str, quant: str = "4bit"):
    """Load (once) and cache the HF model + processor. quant in {'4bit','8bit','bf16'}."""
    global _MODEL, _PROC, _MODEL_ID
    if _MODEL is not None and _MODEL_ID == model_id:
        return
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

    kwargs: Dict[str, Any] = {"dtype": torch.bfloat16, "device_map": "cuda"}
    if quant == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif quant == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    _PROC = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    _MODEL = AutoModelForImageTextToText.from_pretrained(model_id, trust_remote_code=True, **kwargs)
    _MODEL.eval()
    _MODEL_ID = model_id


def _tokenizer():
    return getattr(_PROC, "tokenizer", _PROC)


def hf_query(model_id: str, prompt: str, image_b64: str, max_tokens: int = 30,
             quant: str = "4bit") -> Dict[str, Any]:
    import torch
    from transformers import TextIteratorStreamer

    load_model(model_id, quant)

    img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}],
    }]
    inputs = _PROC.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    inputs = {k: (v.to(_MODEL.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(_tokenizer(), skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(**inputs, max_new_tokens=max_tokens, do_sample=False, streamer=streamer)

    err = {}

    def _run():
        try:
            with torch.no_grad():
                _MODEL.generate(**gen_kwargs)
        except Exception as e:  # surface generate() failures to the caller
            err["e"] = e

    th = threading.Thread(target=_run)
    t0 = time.perf_counter()
    th.start()

    text, t_first, n_tok = "", None, 0
    for chunk in streamer:
        if t_first is None:
            t_first = time.perf_counter()
        text += chunk
        n_tok += 1
    th.join()
    t_end = time.perf_counter()

    if err:
        raise err["e"]
    if t_first is None:  # nothing generated
        t_first = t_end

    prefill_s = max(t_first - t0, 1e-6)
    decode_s = max(t_end - t_first, 1e-6)
    return {
        "response": text.strip(),
        "prompt_eval_duration": int(prefill_s * 1e9),
        "eval_duration": int(decode_s * 1e9),
        "eval_count": n_tok,
    }
