"""
FastAPI LLM Summarization API
=============================

Provides a single endpoint `/summarize` that:
1. Accepts Hebrew text.
2. Translates it to English.
3. Summarizes into 5 bullet points via Phi-3-mini-4k-instruct.
4. Optionally back-translates each bullet into Hebrew.
5. Streams results back to the client in real-time.

Supports generation parameter tuning: temperature, top_p, top_k, repeat_penalty.
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from app.models.translator import Translator
from app.models.text_generator import TextGenerator
from app.models.summarizer import Summarizer
from app.models.streaming import StreamPipeline
from app.schemas import SummarizeRequest, GenerationParams

app = FastAPI(title="LLM Summarization API", version="1.0")

# Initialize models
translator = Translator()
text_generator = TextGenerator()
pipeline = StreamPipeline(translator, text_generator)


def generate_summary(full_translation: str, gen_params: dict, max_tokens: int):
    """
    Generate English summary from translated text using the LLM.
    Streams output continuously.
    """
    yield from pipeline.generate_and_stream(
        Summarizer.build_prompt,
        full_translation,
        max_tokens=max_tokens,
        **gen_params
    )


def back_translate_bullets(full_translation: str, gen_params: dict, max_tokens: int):
    """
    Generate English summary, then translate each bullet to Hebrew as it completes.
    Streams Hebrew bullets immediately.
    """
    bullet_buf = ""
    for chunk in pipeline.generate_and_stream(
        Summarizer.build_prompt,
        full_translation,
        max_tokens=max_tokens,
        **gen_params
    ):
        bullet_buf += chunk
        while "\n" in bullet_buf:
            line, bullet_buf = bullet_buf.split("\n", 1)
            line = line.strip()
            if not line or line == "END SUMMARY":
                continue
            if line.startswith("•"):
                heb = translator.translate(line, src_lang="eng_Latn", tgt_lang="heb_Hebr").strip()
                yield heb + "\n"

    # flush residual bullet if needed
    residual = bullet_buf.strip()
    if residual and residual != "END SUMMARY" and residual.startswith("•"):
        heb = translator.translate(residual, src_lang="eng_Latn", tgt_lang="heb_Hebr").strip()
        yield heb + "\n"


@app.post("/summarize")
async def summarize(payload: SummarizeRequest):
    """
    Summarize Hebrew text into 5 English bullet points.
    Optionally stream Hebrew translation of each bullet.

    Request Body:
        payload (SummarizeRequest):
            - text: Hebrew text to summarize
            - max_tokens: optional max token count
            - back_translate: whether to stream Hebrew bullets
            - generation parameters: temperature, top_p, top_k, repeat_penalty

    Returns:
        StreamingResponse: Yields translated sentences, generation output, and optional back-translated bullets.
    """
    try:
        hebrew_text = payload.text
        max_tokens = payload.max_tokens or 200
        back_translate = payload.back_translate

        # Prepare generation parameters, exclude unset to avoid sending defaults
        gen_params = GenerationParams(
            temperature=payload.temperature,
            top_p=payload.top_p,
            top_k=payload.top_k,
            repeat_penalty=payload.repeat_penalty,
        ).dict(exclude_unset=True)

        # Filter out any None
        gen_params = {k: v for k, v in gen_params.items() if v is not None}

        def stream():
            # --- Phase 1: Hebrew -> English translation ---
            translated_sentences = []
            for t in pipeline.translate_and_stream(hebrew_text):
                yield t
                translated_sentences.append(t.replace("TRANSLATION: ", "").strip())

            full_translation = " ".join(translated_sentences).strip()

            # Divider for clarity in the stream
            yield "\n" + "-"*50 + " GENERATION " + "-"*50 + "\n"

            # Phase 2: Generate summary
            if back_translate:
                # Stream Hebrew bullets directly
                yield from back_translate_bullets(full_translation, gen_params, max_tokens)
            else:
                # Stream English summary
                yield from generate_summary(full_translation, gen_params, max_tokens)

        # StreamingResponse allows real-time output
        return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
