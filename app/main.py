#app/main.py

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from app.models.translator import Translator
from app.models.text_generator import TextGenerator
from app.models.summarizer import Summarizer
from app.models.streaming import StreamPipeline
from app.schemas import GenerationParams

app = FastAPI(title="LLM Summarization API", version="1.0")

# Initialize models
translator = Translator()
text_generator = TextGenerator()
pipeline = StreamPipeline(translator, text_generator)


@app.post("/summarize")
async def summarize(request: Request):
    try:
        data = await request.json()
        hebrew_text = data.get("text")
        if not hebrew_text:
            return JSONResponse({"error": "Missing 'text' field"}, status_code=400)

        max_tokens = data.get("max_tokens", 200)
        params = GenerationParams(**data)  # ignore extra fields automatically
        generation_params = params.dict(exclude_unset=True)

        def stream_generator():
            # Phase 1: Translation
            translated_sentences = []
            for t in pipeline.translate_and_stream(hebrew_text):
                translated_sentences.append(t.replace("TRANSLATION: ", "").strip())
                yield t

            full_translation = " ".join(translated_sentences)

            # Phase 2: Summarization
            yield "\n===SUMMARY START===\n"
            yield from pipeline.generate_and_stream(
                Summarizer.build_prompt,
                full_translation,
                max_tokens=max_tokens,
                **generation_params
            )
            yield "\n"

        return StreamingResponse(stream_generator(), media_type="text/plain; charset=utf-8")

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
