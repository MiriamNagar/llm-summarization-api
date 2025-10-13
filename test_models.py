from models.translator import Translator
from models.summarizer import Summarizer

translator = Translator()
summarizer = Summarizer()

text = "שלום, מה שלומך היום? אני רוצה ללמוד על למידת מכונה."

# Translate Hebrew -> English
english_text = translator.translate(text, src_lang="heb_Hebr", tgt_lang="eng_Latn")
print("Translated text:", english_text)

# Summarize (streaming)
print("\nSummary:")
for chunk in summarizer.summarize(english_text):
    print(chunk, end="", flush=True)
