from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from app.models.translator import Translator
from app.models.summarizer import Summarizer
import re

from bidi.algorithm import get_display
import arabic_reshaper

app = FastAPI(title="LLM Summarization API", version="1.0")

translator = Translator()
summarizer = Summarizer()

@app.post("/summarize")
async def summarize(request: Request):
    """
    Endpoint:
    - Accepts JSON: { "text": "<Hebrew text>", "temperature": 0.7, "top_p": 0.9, "max_tokens": 200 }
    - Translates to English
    - Summarizes via Phi-3-mini model
    - Streams summary chunks
    """
    try:
        data = await request.json()
        hebrew_text = data.get("text")
        temperature = data.get("temperature", 0.0)
        top_p = data.get("top_p", 0.7)
        max_tokens = data.get("max_tokens", 200)
        output_lang = data.get("output_lang", "en")  # 'he' for Hebrew

        if not hebrew_text:
            return JSONResponse({"error": "Missing 'text' field"}, status_code=400)

        # Step 1: Translate to English
        english_text = translator.translate(hebrew_text, src_lang="heb_Hebr", tgt_lang="eng_Latn")

        # Step 2: Summarize and stream back
        def generate_stream():
            for chunk in summarizer.summarize(
                english_text, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stream=True
            ):
                if chunk.strip():
                    yield chunk

        # def generate_stream():
        #     if output_lang.lower() == "he":
        #         return stream_summary_hebrew_realtime(english_text, max_tokens, temperature, top_p)
        #     else:
        #         for chunk in summarizer.summarize(
        #             english_text,
        #             max_tokens=max_tokens,
        #             temperature=temperature,
        #             top_p=top_p,
        #             stream=True
        #         ):
        #             if chunk.strip():
        #                 yield chunk

        return StreamingResponse(generate_stream(), media_type="text/plain; charset=utf-8")
        # return StreamingResponse(
        #     stream_summarization(hebrew_text, max_tokens=max_tokens, temperature=temperature, top_p=top_p),
        #     media_type="text/plain; charset=utf-8",
        # )

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


def stream_summarization(hebrew_text, max_tokens=200, temperature=0.7, top_p=0.9):
    """
    Full streaming pipeline:
    1. Hebrew -> English (sentence by sentence)
    2. Phi-3 generates English summary token-by-token
    3. Translate completed bullet to Hebrew and yield immediately
    """
    # Step 1: Translate Hebrew input sentence by sentence
    english_input = ""
    sentences = translator._split_sentences(hebrew_text, lang="hebrew")
    for sentence in sentences:
        english_sentence = translator.translate(sentence, src_lang="heb_Hebr", tgt_lang="eng_Latn")
        english_input += english_sentence + " "

    # Step 2 + 3: Stream English summary and translate bullets to Hebrew
    buffer = ""
    for token in summarizer.summarize(
        english_input,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True
    ):
        buffer += token
        # Detect bullet boundary
        if "•" in buffer or "\n" in buffer:
            parts = re.split(r"(?:\n|•)", buffer)
            for part in parts[:-1]:
                text = part.strip()
                if text:
                    hebrew_bullet = translator.translate(text, src_lang="eng_Latn", tgt_lang="heb_Hebr")
                    reshaped = arabic_reshaper.reshape(hebrew_bullet)
                    bidi_text = get_display(reshaped)
                    yield "• " + bidi_text + "\n"
            buffer = parts[-1]

    # Any leftover
    if buffer.strip():
        hebrew_bullet = translator.translate(buffer.strip(), src_lang="eng_Latn", tgt_lang="heb_Hebr")
        reshaped = arabic_reshaper.reshape(hebrew_bullet)
        bidi_text = get_display(reshaped)
        yield "• " + bidi_text + "\n"
