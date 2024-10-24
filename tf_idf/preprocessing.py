"""
This module used to clean text or tokenize the sentence to word(tokens)
"""
from dataclasses import dataclass
import re
from pyvi import ViTokenizer
from typing import List
from underthesea import word_tokenize


@dataclass
class TokenizerMethod:
    PYVI = "pyvi"
    UNDERTHESEA = "underthesea"
    WHITESPACE = "whitespace"


def clean_text(
        text,
        lowercase: bool = False,
        special_character=r"[#$%^&@.,:\"]",
) -> str:
    """
    This module use to clean text
    Args:
        text: Text need to clean
        lowercase:
        special_character:

    Returns:

    """
    input_text = re.sub(r"\s{2,}", " ", text)

    if lowercase:
        input_text = input_text.lower()

    if special_character:
        input_text = re.sub(special_character, "", input_text)

    return input_text.strip()


def tokenize(
        input_text,
        token_method: str = TokenizerMethod.PYVI,
        lowercase: bool = True
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
        lowercase:
    Returns:
        List[str]: A list of tokens (words) extracted from the input text.

    Raises:
        ValueError: If the token_method is not one of the specified methods.

    Examples:
        >>> tokenize("Xin chào, tôi tên là Hoa.", TokenizerMethod.PYVI)
        ['Xin', 'chào', ',', 'tôi', 'tên', 'là', 'Hoa', '.']

        >>> tokenize("Xin chào, tôi tên là Hoa.", TokenizerMethod.UNDERTHESEA)
        ['Xin', 'chào', ',', 'tôi', 'tên', 'là', 'ChatGPT', '.']

        >>> tokenize("Hello world!", TokenizerMethod.WHITESPACE)
        ['Hello', 'world!']
    """
    # Tokenize text with pyvi library
    if token_method == TokenizerMethod.PYVI:
        print("input_text: ", input_text)
        tokens = ViTokenizer.tokenize(input_text)
        tokens = tokens.split()
        print("tokens: ", tokens)
    # Tokenize text with underthesea library
    elif token_method == TokenizerMethod.UNDERTHESEA:
        tokens = word_tokenize(input_text)
    # Tokenize text with split builtin method
    elif token_method == TokenizerMethod.WHITESPACE:
        tokens = input_text.split()
    else:
        raise ValueError('Method must be "pyvi", "underthesea", "whitespace"')

    # Lower all token in tokens
    if lowercase:
        tokens = [token.lower() for token in tokens]
    return tokens
