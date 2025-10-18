"""
Pydantic Schemas for FastAPI Endpoints
======================================

Defines request models for the /summarize endpoint, including:
- Generation parameters for LLM control.
- Structured input for summarization requests.
"""

from pydantic import BaseModel, Field, conint, confloat
from typing import Optional


class GenerationParams(BaseModel):
    """
    LLM generation parameters for controlling text output.

    Attributes:
        temperature (float, optional): Sampling temperature (0.0-2.0). Defaults to 0.3.
            Lower values make output more deterministic, higher values increase creativity.
        top_p (float, optional): Cumulative probability for nucleus sampling (0.0-1.0). Defaults to 0.9.
        top_k (int, optional): Top-k sampling; restricts to top k tokens (1-200). Defaults to 40.
        repeat_penalty (float, optional): Penalizes repeated tokens (0.5-2.0). Defaults to 1.1.
    """
    temperature: Optional[confloat(ge=0.0, le=2.0)] = None
    top_p: Optional[confloat(ge=0.0, le=1.0)] = None
    top_k: Optional[conint(ge=1, le=200)] = None
    repeat_penalty: Optional[confloat(ge=0.5, le=2.0)] = None

class SummarizeRequest(GenerationParams):
    """
    Request model for the /summarize endpoint.

    Inherits all generation parameters from `GenerationParams` and adds:
        text (str): Input text in Hebrew to summarize.
        max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 200.
        back_translate (bool, optional): Whether to also stream Hebrew translation of each bullet. Defaults to False.
    """
    text: str = Field(..., description="Input text in Hebrew")
    max_tokens: Optional[conint(ge=32, le=1024)] = 200
    back_translate: Optional[bool] = Field(False, description="Also stream Hebrew per bullet")
