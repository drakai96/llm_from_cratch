from typing import List

from pydantic import BaseModel, Field


class Documents(BaseModel):
    docs: List[str,] = Field(description="List of sequence")
