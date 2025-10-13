# app/models/translator.py
"""
Translator class using facebook/nllb-200-distilled-600M
Robustly handles Hebrew (or other languages) and ensures all sentences are translated.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from bidi.algorithm import get_display
import arabic_reshaper
from typing import List
import re


class Translator:
    def __init__(self, model_name="facebook/nllb-200-distilled-600M"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def _split_sentences(self, text: str, lang="hebrew") -> List[str]:
        """
        Split text into sentences.
        For unsupported languages (like Hebrew), uses punctuation-based splitting.
        """
        sentences = re.split(r'(?<=[\.\!\?])\s+', text.strip())
        return [s for s in sentences if s]

    def translate(self, text: str, src_lang="heb_Hebr", tgt_lang="eng_Latn") -> str:
        """
        Translate the full text sentence by sentence.
        """
        sentences = self._split_sentences(text, lang='hebrew' if src_lang == 'heb_Hebr' else 'english')
        translated_sentences = []

        for sentence in sentences:
            self.tokenizer.src_lang = src_lang
            inputs = self.tokenizer(sentence, return_tensors="pt", padding=True)
            forced_bos_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
            if forced_bos_id is None:
                raise ValueError(f"Unknown target language code: {tgt_lang}")

            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_id,
                max_new_tokens=200
            )
            translated = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            translated_sentences.append(translated)

        return " ".join(translated_sentences)

    def translate_for_display(self, text: str, src_lang="heb_Hebr", tgt_lang="eng_Latn") -> str:
        """
        Translate and reshape for proper console display (e.g., Hebrew right-to-left)
        """
        logical_text = self.translate(text, src_lang, tgt_lang)
        reshaped = arabic_reshaper.reshape(logical_text)
        bidi_text = get_display(reshaped)
        return bidi_text
