"""
Streaming Pipeline Module
=========================

Provides a generic streaming interface for:
1. Translating text between languages.
2. Generating output from a text generation model.
3. Translating generated output back to the original language.

This module is designed to work with:
- `Translator`
- `TextGenerator`
"""

from typing import Generator, Callable, Iterable
from app.models.translator import Translator
from app.models.text_generator import TextGenerator


class StreamPipeline:
    """
    Orchestrates streaming translation and text generation.

    Attributes:
        translator (Translator): Instance of the Translator class.
        generator (TextGenerator): Instance of the TextGenerator class.
    """

    def __init__(self, translator: Translator, generator: TextGenerator):
        """
        Initialize the streaming pipeline with a translator and a generator.

        Args:
            translator (Translator): Handles multilingual translation.
            generator (TextGenerator): Handles text generation.
        """
        self.translator = translator
        self.generator = generator

    def translate_and_stream(
        self, text: str, src_lang: str = "heb_Hebr", tgt_lang: str = "eng_Latn"
    ) -> Iterable[str]:
        """
        Translate input text sentence by sentence and stream output.

        Args:
            text (str): Input text to translate.
            src_lang (str): Source language code (default "heb_Hebr").
            tgt_lang (str): Target language code (default "eng_Latn").
        Yields:
            str: Each translated sentence prefixed with 'TRANSLATION:'.
        """
        sentences = self.translator.split_sentences(text)
        for sentence in sentences:
            translated = self.translator.translate(sentence, src_lang, tgt_lang).strip()
            yield f"TRANSLATION: {translated}\n"

    def generate_and_stream(
        self,
        prompt_builder: Callable[[str], str],
        input_text: str,
        max_tokens: int = 200,
        **generation_params
    ) -> Iterable[str]:
        """
        Generate output from the LLM in streaming mode using a prompt builder.

        Args:
            prompt_builder (Callable[[str], str]): Function that builds a prompt from input text.
            input_text (str): The text to feed into the prompt builder.
            max_tokens (int): Maximum number of tokens to generate (default 200).
            **generation_params: Additional parameters for text generation
                (e.g., temperature, top_p).

        Yields:
            str: Generated text chunks as they are produced by the model.
        """
        prompt = prompt_builder(input_text)
        yield from self.generator.generate(prompt, stream=True, max_tokens=max_tokens, **generation_params)

    def translate_stream(
        self, text: str, src_lang: str = "eng_Latn", tgt_lang: str = "heb_Hebr"
    ) -> Iterable[str]:
        """
        Translate generated English text back to the original language (e.g., Hebrew)
        in a streaming fashion.

        Args:
            text (str): Input text to translate.
            src_lang (str): Source language code (default "eng_Latn").
            tgt_lang (str): Target language code (default "heb_Hebr").
        Yields:
            str: Translated sentences.
        """
        sentences = self.translator.split_sentences(text)
        for sentence in sentences:
            translated = self.translator.translate(sentence, src_lang, tgt_lang).strip()
            yield translated + "\n"
