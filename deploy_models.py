"""
deploy_models.py

Part 1: Model Deployment – Verification Script
===============================================

Demonstrates loading and verifying the models for the LLM Summarization API

Purpose:
- Ensure both models load correctly without errors.
- Verify that streaming generation and translation produce coherent outputs.
- Serve as a reference/test before integrating into FastAPI.
"""

import logging
from app.models.text_generator import TextGenerator
from app.models.translator import Translator

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
)

logger = logging.getLogger(__name__)

def verify_text_generator():
    """Verify that the Phi-3-mini-4k-instruct model loads and generates text (streaming)."""
    logger.info("Loading Phi-3-mini-4k-instruct model (via llama.cpp)...")
    try:
        gen = TextGenerator(model_path="Phi-3-mini-4k-instruct-q4.gguf")
        prompt = (
            "### Instruction:\n"
            "Write ONE clear, concise sentence describing what machine learning is.\n"
            "Output ONLY the sentence. Do NOT include any extra tokens or labels.\n"
            "End your output with the phrase 'END SUMMARY'.\n"
            "\n### Input:\n"
            "(no input needed)\n\n### Response:\n"
        )
        logger.info("Running streaming inference...")
        print("-" * 50)
        streamed_output = []
        for chunk in gen.generate(prompt, stream=True, max_tokens=50):
            if chunk:
                print(chunk, end="", flush=True)
                streamed_output.append(chunk)
        print()  # newline
        print("-" * 50)
        logger.info("Streaming generation ended.")
        final_text = "".join(streamed_output).strip()
        logger.info("Final generated text: %s", final_text)
    except Exception as e:
        logger.exception("Failed to verify text generator: %s", e)


def verify_translator():
    """Verify that the translation model loads and translates correctly."""
    logger.info("Loading facebook/nllb-200-distilled-600M translation model...")
    try:
        tr = Translator()
        heb_text = "שלום, זהו מבחן קצר של מערכת התרגום."
        logger.info("Translating Hebrew → English...")
        eng = tr.translate(heb_text, src_lang="heb_Hebr", tgt_lang="eng_Latn")
        logger.info("English translation: %s", eng)
        logger.info("Translating English → Hebrew...")
        back = tr.translate(eng, src_lang="eng_Latn", tgt_lang="heb_Hebr")
        logger.info("Back translation: %s", back)
        logger.info("Translation verified successfully.\n")
    except Exception as e:
        logger.exception("Failed to verify translator: %s", e)

if __name__ == "__main__":
    verify_text_generator()
    print("="*80)
    verify_translator()
