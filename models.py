"""Thin wrapper over local Moondream-2 via Ollama.

Ollama keeps the model resident between calls, so per-call latency is low after
the first request. Temperature 0 for deterministic 'bare'; >0 for sampling
scaffolds.
"""
from __future__ import annotations
import io
from typing import Optional
import ollama
from PIL import Image


class Moondream:
    """Thin Ollama wrapper. The class name is historical — pass any
    multimodal Ollama model via `model_name`. Defaults are tuned for a
    16 GB Mac: small context and short keep_alive to bound resident size.
    """

    name = "moondream"

    def __init__(
        self,
        model_name: str = "moondream",
        num_ctx: int = 1024,
        keep_alive: str = "30s",
    ):
        self.model_name = model_name
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive

    def query(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> str:
        msg = {"role": "user", "content": prompt}
        if image is not None:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            msg["images"] = [buf.getvalue()]
        resp = ollama.chat(
            model=self.model_name,
            messages=[msg],
            keep_alive=self.keep_alive,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
                "stop": ["\n\n\n"],
            },
        )
        return resp["message"]["content"]
