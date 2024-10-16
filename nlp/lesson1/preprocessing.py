"""
This module used to clean text or tokenize the sentence to word(tokens)
"""

import re
from typing import List, Literal

from pyvi import ViTokenizer
from underthesea import word_tokenize


def clean_text(
        input_text,
        lower: bool = True,
        white_space: bool = True,
        special_character=r"[#$%^&@.]",
) -> str:
    """
    This module use to clean text
    Args:
        input_text: Text need to clean
        lower:
        white_space:
        special_character:

    Returns:

    """
    if white_space:
        input_text = re.sub(r"\s{2,}", " ", input_text)

    if lower:
        input_text = input_text.lower()

    if special_character:
        input_text = re.sub(special_character, "", input_text)

    return input_text.strip()


def tokenize_text(
        input_text, token_method: Literal["pyvi", "underthesea", "split"] = "pyvi"
) -> List[str,]:
    """
    Tokenizing the input text into words (tokens).

    Args:
        input_text (str): The text that needs to be tokenized.
        It can be a single sentence or a larger body of text.
        token_method (Literal["pyvi", "underthesea", "split"]): The method to use for tokenization.
            - "pyvi": Uses the PyVi library for Vietnamese tokenization.
            - "underthesea": Uses the Underthesea library for Vietnamese tokenization.
            - "split": Uses simple string splitting based on whitespace.

    Returns:
        List[str]: A list of tokens (words) extracted from the input text.

    Raises:
        ValueError: If the token_method is not one of the specified methods.

    Examples:
        >>> tokenize_text("Xin chào, tôi tên là Hoa.", "pyvi")
        ['Xin', 'chào', ',', 'tôi', 'tên', 'là', 'Hoa', '.']

        >>> tokenize_text("Xin chào, tôi tên là Hoa.", "underthesea")
        ['Xin', 'chào', ',', 'tôi', 'tên', 'là', 'ChatGPT', '.']

        >>> tokenize_text("Hello world!", "split")
        ['Hello', 'world!']
    """
    match token_method:
        # Tokenize text with pyvi library
        case "pyvi":
            tokens = ViTokenizer.tokenize(input_text)
            return tokens.split()
        # Tokenize text with underthesea library
        case "underthesea":
            tokens = word_tokenize(input_text)
            return tokens
        # Tokenize text with split builtin method
        case "split":
            return input_text.split()
        case _:
            raise ValueError('Method must be "pyvi", "underthesea", "split"')


if __name__ == "__main__":
    text = " Xin Chào mình là    Nhậm . $%&$#&$"
    text_clean = clean_text(input_text=text, white_space=True, lower=True, special_character=r"[#$%^&@.]")
    print(text_clean)
    method_token = ["pyvi", "underthesea", "split"]
    for method in method_token:
        print("token with: ", method)
        tokens_ = tokenize_text(input_text=text_clean, token_method=method)
        print(tokens_)
