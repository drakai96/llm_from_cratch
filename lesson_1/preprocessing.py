"""
This module used to clean text or tokenize the sentence to word(tokens)
"""

import re
from enum import Enum
from pyvi import ViTokenizer
from typing import List
from underthesea import word_tokenize


class TokenizerMethod(Enum):
    PYVY = "pyvi"
    UNDERTHESEA = "underthesea"
    SPLIT = "split"


def clean_text(
        input_text,
        lower: bool = False,
        white_space: bool = True,
        special_character=r"[#$%^&@.,:\"]",
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
        input_text, token_method: TokenizerMethod = "pyvi",
        is_lower: bool = True
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
        is_lower:
    Returns:
        List[str]: A list of tokens (words) extracted from the input text.

    Raises:
        ValueError: If the token_method is not one of the specified methods.

    Examples:
        >>> tokenize_text("Xin chào, tôi tên là Hoa.", TokenizerMethod.PYVY)
        ['Xin', 'chào', ',', 'tôi', 'tên', 'là', 'Hoa', '.']

        >>> tokenize_text("Xin chào, tôi tên là Hoa.", TokenizerMethod.UNDERTHESEA)
        ['Xin', 'chào', ',', 'tôi', 'tên', 'là', 'ChatGPT', '.']

        >>> tokenize_text("Hello world!", TokenizerMethod.SPLIT)
        ['Hello', 'world!']
    """
    # Tokenize text with pyvi library
    if token_method == TokenizerMethod.PYVY:
        print("input_text: ", input_text)
        tokens = ViTokenizer.tokenize(input_text)
        tokens = tokens.split()
        print("tokens: ", tokens)
    # Tokenize text with underthesea library
    elif token_method == TokenizerMethod.UNDERTHESEA:
        tokens = word_tokenize(input_text)
    # Tokenize text with split builtin method
    elif token_method == TokenizerMethod.SPLIT:
        tokens = input_text.split()
    else:
        raise ValueError('Method must be "pyvi", "underthesea", "split"')

    # Lower all token in tokens
    if is_lower:
        tokens = [token.lower() for token in tokens]
    return tokens
