#app/models/streaming.py
"""
Generic streaming pipeline:
1. Translate text if needed.
2. Generate output with TextGenerator.
"""

from typing import Generator, Callable
from app.models.translator import Translator
from app.models.text_generator import TextGenerator


class StreamPipeline:
    def __init__(self, translator: Translator, generator: TextGenerator):
        self.translator = translator
        self.generator = generator

    def translate_and_stream(
        self, text: str, src_lang="heb_Hebr", tgt_lang="eng_Latn"
    ) -> Generator[str, None, str]:
        """
        Translate sentence-by-sentence and stream each sentence.
        Returns full translated text after streaming.
        """
        sentences = self.translator._split_sentences(text)
        translated_parts = []

        for i, sentence in enumerate(sentences, start=1):
            translated = self.translator.translate(sentence, src_lang, tgt_lang).strip()
            translated_parts.append(translated)
            print(f"[translation {i}] {translated}")  # server log
            yield f"TRANSLATION: {translated}\n"

        full_translation = " ".join(translated_parts).strip()
        return full_translation

    def generate_and_stream(
        self,
        prompt_builder: Callable[[str], str],
        input_text: str,
        max_tokens: int = 200,
        **generation_params
    ) -> Generator[str, None, None]:
        """
        Generic generation stream.
        `prompt_builder` defines what to ask the model (e.g., summarization, Q&A, etc.)
        """
        prompt = prompt_builder(input_text)
        yield from self.generator.generate(prompt, stream=True, max_tokens=max_tokens, **generation_params)
