# app/main.py
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from app.models.translator import Translator
from app.models.summarizer import Summarizer

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
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 0.9)
        max_tokens = data.get("max_tokens", 200)

        if not hebrew_text:
            return JSONResponse({"error": "Missing 'text' field"}, status_code=400)

        # Split into sentences (your translator has _split_sentences)
        sentences = translator._split_sentences(hebrew_text, lang="hebrew")

        def generate_stream():
            # Phase 1: streamed translation (sentence-by-sentence)
            english_parts = []
            for i, sentence in enumerate(sentences, start=1):
                translated = translator.translate(sentence, src_lang="heb_Hebr", tgt_lang="eng_Latn")
                translated = translated.strip()
                english_parts.append(translated)

                # print translation to server console for verification
                print(f"[translation {i}] {translated}")

                # yield translation line to client immediately
                yield f"TRANSLATION: {translated}\n"

            # Join accumulated english
            full_english = " ".join(english_parts).strip()

            # Small separator so the client knows summary will start
            yield "\n===SUMMARY START===\n"

            # Phase 2: summarizer called once; stream tokens as they arrive
            for token in summarizer.summarize(
                full_english,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True
            ):
                # token is a small string (piece of text). yield immediately.
                yield token

            # final newline to mark end-of-stream nicely
            yield "\n"

        # use text/plain streaming so simple curl/clients can see the stream
        return StreamingResponse(generate_stream(), media_type="text/plain; charset=utf-8")

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
