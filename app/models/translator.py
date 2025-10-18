"""
Translator Module
=================
Provides a simple and production-ready interface for performing machine translation
using the facebook/nllb-200-distilled-600M model from Hugging Face.

This module defines the `Translator` class, which can:
- Translate between any supported source and target languages.
- Handle multi-sentence text via punctuation-based segmentation.
- Be easily integrated with FastAPI or other pipelines.

Example:
    >>> translator = Translator()
    >>> translator.translate("שלום עולם", src_lang="heb_Hebr", tgt_lang="eng_Latn")
    'Hello world'
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re
from typing import List


class Translator:
    """
    A wrapper around the facebook/nllb-200-distilled-600M translation model.

    This class supports translating between a wide variety of language pairs,
    including Hebrew ↔ English, using Hugging Face Transformers.

    Attributes:
        model_name (str): The model identifier used to load the NLLB model.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for encoding/decoding.
        model (transformers.PreTrainedModel): The underlying translation model.
    """

    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M"):
        """
        Initialize the Translator instance.

        Loads the tokenizer and model weights into memory. This may take a few seconds
        the first time, as Hugging Face downloads the model.

        Args:
            model_name (str): The name or path of the model to load.
                Defaults to "facebook/nllb-200-distilled-600M".
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """
        Split the input text into separate sentences.

        This function uses punctuation-based splitting, which is suitable for
        most languages (including Hebrew, which may not use capitalization or
        clear sentence delimiters).

        Args:
            text (str): The input text to split.

        Returns:
            List[str]: A list of individual sentences.
        """
        parts = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
        return [p.strip() for p in parts if p]

    @torch.no_grad()
    def translate(
        self,
        text: str,
        src_lang: str = "heb_Hebr",
        tgt_lang: str = "eng_Latn",
        max_new_tokens: int = 400,
        temperature: float = None,
        top_p: float = None,
    ) -> str:
        """
        Translate text from one language to another using the NLLB model.

        Args:
            text (str): The input text to translate.
            src_lang (str): The source language code.
                Defaults to "heb_Hebr" (Hebrew).
            tgt_lang (str): The target language code.
                Defaults to "eng_Latn" (English).
            max_new_tokens (int): The maximum number of new tokens to generate.
                Defaults to 400.
            temperature (float): Sampling temperature. If `None`, deterministic
                (greedy) decoding is used.
            top_p (float): Nucleus sampling probability. If `None`, deterministic
                decoding is used.

        Returns:
            str: The translated text.

        Raises:
            ValueError: If the target language code is unknown or unsupported.

        Example:
            >>> translator = Translator()
            >>> translator.translate("מה שלומך?", src_lang="heb_Hebr", tgt_lang="eng_Latn")
            'How are you?'
        """
        if not text.strip():
            return ""

        # Set source language
        self.tokenizer.src_lang = src_lang

        # Encode text into model input format
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        # Validate target language code
        forced_bos_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        if forced_bos_id is None or forced_bos_id == self.tokenizer.unk_token_id:
            raise ValueError(f"Unknown target language code: {tgt_lang}")

        # Determine whether to use sampling (temperature/top_p) or deterministic decoding
        use_sampling = temperature is not None or top_p is not None

        # Generate translation
        generation_kwargs = {
            **inputs,
            "forced_bos_token_id": forced_bos_id,
            "max_new_tokens": max_new_tokens,
        }

        if use_sampling:
            generation_kwargs.update({
                "do_sample": True,
                "temperature": temperature or 1.0,
                "top_p": top_p or 1.0,
            })

        generated_tokens = self.model.generate(**generation_kwargs)

        # Decode and clean up the translated output
        translation = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translation.strip()
