"""
Summarizer Module
=================
Provides a prompt-building utility for text summarization.  
The module does not generate text itself, but prepares prompts that can be fed into a text generation model (like Phi-3-mini-4k-instruct).

Example:
    >>> from app.models.summarizer import Summarizer
    >>> prompt = Summarizer.build_prompt("Your input text here.")
"""

class Summarizer:
    """
    Prompt builder for summarization.

    This class is responsible for constructing a professional, structured prompt
    instructing the LLM to summarize text into exactly 5 concise, human-readable bullet points.
    It ensures that the output is suitable for translation back into Hebrew or other languages.
    """

    @staticmethod
    def build_prompt(text: str) -> str:
        """
        Construct a structured prompt for summarizing the given text.

        Args:
            text (str): The input text to summarize.
        Returns:
            str: A string containing the full prompt to pass to the LLM.
        """
        return (
            "You are a professional English writer and summarizer.\n"
            "Summarize the following text into exactly 5 concise, natural bullet points.\n"
            "- Focus on meaning and intention rather than literal phrasing.\n"
            "- Use clear, complete sentences suitable for direct translation to Hebrew. Avoid idioms or complex phrasing.\n"
            "- Each bullet should stand alone as a complete, human-readable sentence.\n"
            "- Use simple, fluent English and vary structure.\n"
            "- Start each bullet with '•' (U+2022) and place each bullet on a new line.\n"
            "- Output ONLY the 5 bullets, then the phrase 'END SUMMARY'.\n"
            "- Do not repeat or restate information.\n"
            f"\nText:\n{text}\n\nOutput:\n"
        )
