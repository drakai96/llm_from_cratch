"""
Create a class which use to clean text and split sentence to word
"""

import re
from typing import List, Optional

from pyvi import ViTokenizer


class TextPreprocessing:
    """
    Cleaning and tokenizing text
    """
    def __init__(
        self,
        white_space: bool = True,
        lower: bool = True,
        upper: bool = False,
        regex: Optional[str] = r"[.%$#@!]",
    ) -> None:
        """
        Init class
        Args:
            white_space: True if replace more than 2 white space to one (default value : True)
            lower: True if lower all word in string (default value : True)
            upper: True if upper all word in string (default value : False)
            regex(Nullable): Remove all special character in string (default value : r"[.%$#@!]")
        """
        self.white_space = white_space
        self.lower = lower
        self.upper = upper
        self.regex = regex
        if upper and lower:
            raise ValueError("Upper and lower can not be True in same time")

    def clean_text(self, text: str) -> str:
        """
        This function use to clean text
        Args:
            text: Input text

        Returns:
            Text was cleaned
        """
        if self.white_space:
            text = re.sub(r"\s+", " ", text).strip()
        if self.regex:
            text = re.sub(self.regex, "", text)
        if self.lower:
            text = text.lower()
        elif self.upper:
            text = text.upper()
        return text

    def tokenize_sentence_to_words(self, text, use_pyvi: bool = True) -> List[str]:
        """
        Split sentence to word
        Args:
            text: Text need to split
            use_pyvi: True is use pyvi library (default = True).
                  False is use split method
        Returns:
            List of word
        """
        text = self.clean_text(text)

        # Use pyvi library to extract text
        if use_pyvi:
            words = ViTokenizer.tokenize(text)
            words = words.split()
            for i in range(len(words)):
                words[i] = words[i].replace("_", " ")
            return words

        # Else case
        words = text.split()
        return words


if __name__ == "__main__":

    text_test = "nguyen van   nham#@$@"
    text_preproccessing = TextPreprocessing()
    words_ = text_preproccessing.tokenize_sentence_to_words(text_test)
    print(words_)
