"""
TextGenerator class using Phi-3-mini-4k-instruct (via llama_cpp).
General-purpose text generation interface with streaming support.
"""

from llama_cpp import Llama
import os
from typing import Generator


class TextGenerator:
    """
    Generic text generation class.
    Responsible only for interfacing with the LLM.
    """

    def __init__(self, model_path: str = "Phi-3-mini-4k-instruct-q4.gguf",
                 n_ctx: int = 4096,
                 n_threads: int = 8):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads, verbose=False)

    def generate(self, prompt: str, stream: bool = True, max_tokens: int = 200, **generation_params):
        # Force only the parameters that must always be set
        params = {"max_tokens": max_tokens, "stop": ["END SUMMARY"], **generation_params}

        if stream:
            buffer = ""
            for output in self.model.create_completion(prompt=prompt, stream=True, **params):
                token = output["choices"][0]["text"]
                if not token:
                    continue

                buffer += token

                # If the stop phrase is forming, don't yield anything until we confirm it
                if "END SUMMARY" in buffer:
                    clean = buffer.split("END SUMMARY")[0].rstrip()
                    if clean:
                        yield clean + "\nEND SUMMARY\n"
                    break
                else:
                    yield token
        else:
            output = self.model.create_completion(prompt=prompt, **params)
            text = output["choices"][0]["text"]
            if "END SUMMARY" in text:
                text = text.split("END SUMMARY")[0].rstrip() + "\nEND SUMMARY\n"
            yield text
