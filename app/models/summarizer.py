# app/models/summarizer.py
class Summarizer:
    """
    Prompt builder for summarization. Does not generate itself.
    """

    @staticmethod
    def build_prompt(text: str) -> str:
        return (
            "You are a professional English writer and summarizer.\n"
            "Summarize the following text into exactly 5 concise, natural bullet points.\n"
            "- Focus on meaning and intention rather than literal phrasing.\n"
            "- Each bullet should stand alone as a complete, human-readable sentence.\n"
            "- Use simple, fluent English and vary structure.\n"
            "- Start each bullet with '•' (U+2022) and place each bullet on a new line.\n"
            "- Do not repeat or restate information.\n"
            "- Do not include anything before or after the 5 bullets.\n"
            "- End the list with the phrase 'END SUMMARY'.\n"
            f"\nText:\n{text}\n\nOutput:\n"
        )