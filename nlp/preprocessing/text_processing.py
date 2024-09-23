import re
from typing import List

from pydantic import Field


class CleanDocument:
    documents: List[str] = Field(description="List of doc string")
    text: str = Field(description="Text need to clean")
    __AtriWhiteSpace__: bool = True
    __AtriUpper__: bool = False
    __AtriLower__: bool = False
    __AtriClean__: bool = Field(default=True, description="Checking clean documents")

    def __init__(self):
        pass

    def clean_text(
        self,
        long_text: text,
        white_space: __AtriWhiteSpace__,
        upper: __AtriUpper__,
        lower: __AtriLower__,
        special_character: text = r"[\#\@\$\%\,\^\(\)\.]+",
    ) -> str:
        """

        Args:
            long_text: String
                Mean input text - sentence
            white_space: Boolean
                Mean True if clean many white space in a sentence
            upper: Boolean
                Mean True if upper all word in a sentence
            lower: Boolean
                Mean True if lower all word in a sentence
            special_character: String, default = r"[\#\@\$\%\,\^\(\)\.]+"
                Mean clear all character in special_character,
        Returns:
            The cleaning sentence
        """
        if not isinstance(long_text, str) or not long_text:
            raise "Invalid input text. Input must be str"

        if white_space:
            long_text = re.sub(r"\s{2,}", " ", long_text)
        if lower:
            long_text = long_text.lower()
        if upper:
            long_text = long_text.upper()
        if special_character:
            long_text = re.sub(special_character, "", long_text)
        return long_text

    def clean_corpus(self, documents: documents) -> documents:
        """

        Args:
            documents: List of string
                Mean the list of sentence need to clean

        Returns:
                List of the cleaning sentences
        """
        clean_docs = [
            self.clean_text(doc, white_space=True, lower=True, upper=False)
            for doc in documents
        ]
        return clean_docs
