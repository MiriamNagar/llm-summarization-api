#app/schemas.py
from pydantic import BaseModel
from typing import Optional, List

class GenerationParams(BaseModel):
    temperature: Optional[float] = 0.7       # sampling temperature
    top_p: Optional[float] = 0.9            # nucleus sampling
    top_k: Optional[int] = 40               # top-k sampling
    repeat_penalty: Optional[float] = 1.1   # discourage repetitions
    do_sample: Optional[bool] = True        # enable sampling