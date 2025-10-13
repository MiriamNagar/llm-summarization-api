# app/models/summarizer.py
"""
Summarizer class using Phi-3-mini-4k-instruct (via llama_cpp).
Supports streaming responses for FastAPI.
"""

from llama_cpp import Llama
import os
from typing import Generator

class Summarizer:
    def __init__(self, model_path="Phi-3-mini-4k-instruct-q4.gguf", n_ctx=4096, n_threads=8):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads, verbose=False)

    def summarize(self, text: str, max_tokens=200, temperature=0.7, top_p=0.9, stream=True) -> Generator[str, None, None]:
        """
        Generate a summary in bullet points from the input text.
        Returns a generator for streaming.

        Args:
            text (str): Input text to summarize.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature.
            top_p (float): Nucleus sampling probability.
            stream (bool): Whether to stream results or not.

        Yields:
            str: Each chunk of the generated text.
        """
        prompt = (
            "Summarize the following text into exactly 5 bullet points.\n"
            "- Use '•' at the start of each bullet.\n"
            "- Do not add greetings or extra text.\n"
            f"Text:\n{text}"
        )
        messages = [{"role": "user", "content": prompt}]
        # messages = [{
        #     "role": "user",
        #     "content": f"Summarize the technical content of the following text into 5 bullet points. "
        #             f"Ignore greetings or conversational phrases.\n\n{text}"
        # }]

        for output in self.model.create_chat_completion(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream
        ):
            # yield output['choices'][0]['delta'].get('content', '')\
            token = output['choices'][0]['delta'].get('content', '')
            if token:
                yield token
