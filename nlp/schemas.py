from pydantic import BaseModel, Field
from typing import List


class Documents(BaseModel):
    docs: List[str,] = Field(description="List of sequence")
