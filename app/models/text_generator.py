"""
TextGenerator Module
====================
Provides a clean interface for text generation using the Phi-3-mini-4k-instruct
model via llama_cpp. Supports both streaming and non-streaming generation.

The module defines the `TextGenerator` class, which:
- Loads a local GGUF LLaMA model.
- Generates text given a prompt.
- Streams output in real-time or returns the full text at once.
- Handles custom generation parameters (max_tokens, stop tokens, etc.).

Example:
    >>> generator = TextGenerator()
    >>> for chunk in generator.generate("Summarize the following text:", stream=True):
    ...     print(chunk, end="")
"""

from llama_cpp import Llama
import os
from typing import Generator, Iterable, Dict


class TextGenerator:
    """
    Wrapper class for general-purpose text generation using Phi-3-mini-4k-instruct.

    Attributes:
        model (llama_cpp.Llama): The loaded LLaMA model instance.
    """

    def __init__(
        self,
        model_path: str = "Phi-3-mini-4k-instruct-q4.gguf",
        n_ctx: int = 4096,
        n_threads: int = 8,
        n_batch: int = 256,
    ):
        """
        Initialize the text generator by loading the model into memory.

        Args:
            model_path (str): Path to the GGUF model file. Defaults to "Phi-3-mini-4k-instruct-q4.gguf".
            n_ctx (int): Maximum context length. Defaults to 4096.
            n_threads (int): Number of threads for inference. Defaults to 8.
            n_batch (int): Number of tokens to process in a batch. Defaults to 256.

        Raises:
            FileNotFoundError: If the model file does not exist at the given path.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=n_batch,
            verbose=False,
        )

    def generate(
        self,
        prompt: str,
        stream: bool = True,
        max_tokens: int = 200,
        stop: str = "END SUMMARY",
        **generation_params: Dict,
    ) -> Iterable[str]:
        """
        Generate text from a prompt, optionally streaming output in real-time.

        Args:
            prompt (str): The text prompt to generate from.
            stream (bool): If True, yields chunks as they are generated. Defaults to True.
            max_tokens (int): Maximum number of tokens to generate. Defaults to 200.
            stop (str): Stop token that ends generation. Defaults to "END SUMMARY".
            **generation_params (dict): Additional parameters for text generation
                (e.g., temperature, top_p).

        Yields:
            str: Generated text chunks if streaming, or full text if `stream=False`.

        Example:
            >>> generator = TextGenerator()
            >>> for chunk in generator.generate("Summarize:", stream=True):
            ...     print(chunk, end="")
        """
        params = {
            "max_tokens": max_tokens,
            "stop": [stop],
            **generation_params,
        }

        if stream:
            pending = ""
            for out in self.model.create_completion(prompt=prompt, stream=True, **params):
                tok = out["choices"][0]["text"]
                if not tok:
                    continue
                pending += tok

                # Emit text up to stop token
                if stop in pending:
                    before = pending.split(stop)[0]
                    if before:
                        yield before
                    yield f"\n{stop}\n"
                    return

                # Keep trailing characters in case stop splits across tokens
                keep = len(stop) - 1
                if len(pending) > keep:
                    emit = pending[:-keep] if keep > 0 else pending
                    yield emit
                    pending = pending[-keep:] if keep > 0 else ""

            if pending:
                yield pending
        else:
            out = self.model.create_completion(prompt=prompt, **params)
            text = out["choices"][0]["text"]
            if stop in text:
                text = text.split(stop)[0].rstrip() + f"\n{stop}\n"
            yield text
