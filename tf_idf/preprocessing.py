"""
This module used to clean text or tokenize the sentence to word(tokens)
"""

import re
from dataclasses import dataclass
from typing import List, Iterable, Literal

from pyvi import ViTokenizer
from underthesea import word_tokenize


@dataclass
class TokenizerMethod:
    PYVI = "pyvi"
    UNDERTHESEA = "underthesea"
    WHITESPACE = "whitespace"


TOKEN_METHOD = ["pyvi", "underthesea", "whitespace"]


@dataclass
class StopwordLanguage:
    VN = "vi"
    ENG = "eng"


stopword_language = Literal["vi", "eng", None]

stopword_mapping_file = {"vi": "tf_idf/dataset/vietnamese-stopwords.txt", "eng": ...}


def clean_text(
    text,
    lowercase: bool = False,
    special_character=r"[#$%^&@.,:\"\'\(\)]",
) -> str:
    """
    Cleaning text
    Args:
        text: Text need to clean
        lowercase: Lower text if True
        special_character: Remove special text

    Returns:
        Text cleaned
    """
    input_text = re.sub(r"\s{2,}", " ", text)

    if lowercase:
        input_text = input_text.lower()

    if special_character:
        input_text = re.sub(special_character, "", input_text)

    return input_text.strip()


def tokenize(
    input_text, token_method: str = TokenizerMethod.PYVI, lowercase: bool = True
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


def remove_stopwords(tokens, lang: stopword_language = "vi") -> List[str]:
    """
    Remove unimportant tokens(stopwords)
    Args:
        tokens: Tokens need to remove (stopwords)
        lang: Language
    Returns:
        Array of tokens
    """

    # Get stopwords collection
    if lang is None:
        return tokens
    stopword_file = stopword_mapping_file.get(lang)
    if not stopword_file:
        raise ValueError("Language is not setting")
    with open(file=stopword_file, mode="r", encoding="utf-8") as f:
        stop_words_: list = f.readlines()
        stop_words = [word.replace("\n", "").strip() for word in stop_words_]
    stop_words = set(stop_words)

    # Remove stopwords
    tokens_remove_stopword = []
    for token in tokens:
        if token not in stop_words:
            tokens_remove_stopword.append(token)
    return tokens_remove_stopword


def preprocess_tokens(
    documents: Iterable[str,],
    method: str = TokenizerMethod.UNDERTHESEA,
    lowercase: bool = False,
    special_character: str = r"[?!#$%^&@.,:\"]",
    stopword: stopword_language = None,
) -> List[List[str,]]:
    """
    Clean and tokenize iterable of documents into matrix of tokens
    Args:
        documents: Iterable
            - iterable of documents (collection documents)
        method: str - Method tokenizing, default = "underthesea"
            - underthesea method if "underthesea"
            - pyvi method if "pyvi"
            - whitespace method if "whitespace"
        lowercase: Lower all documents if lowercase = True, default = false
        special_character: str, default = [#$%^&@.,:\"]
            Remove special character in special_character
        stopword: Remove ("vi", "end") stopwords if not None
    Returns:
        Matrix of tokens
    Example:
        >>> documents_ = [
        ...     "Xin chào tôi là Quốc Hưng mê gái",
        ...     "Tôi là Nhậm, hãy đưa tiền cho tôi"
        ... ]
        >>> preprocess_tokens(documents,
        ...                 method="whitespace",
        ...                 lowercase=True,
        ...                 special_character=r"[#@]",
        ...                 stopword="vi")
        [['xin', 'chào', 'thế', 'giới', 'đây', 'là', 'một', 'tài', 'liệu', 'thử', 'nghiệm'],
         ['tài', 'liệu', 'khác', 'có', 'ký', 'tự', 'đặc', 'biệt', 'số', '123']]

    """
    matrix_tokens = []
    for doc in documents:
        # Clean text
        doc_cleaning = clean_text(
            doc, lowercase=False, special_character=special_character
        )
        # Tokenizing text
        tokens = tokenize(doc_cleaning, lowercase=lowercase, token_method=method)

        # Remove stopwords
        tokens = remove_stopwords(tokens=tokens, lang=stopword)
        matrix_tokens.append(tokens)
    return matrix_tokens


if __name__ == "__main__":
    documents_ = [
        "Xin chào tôi là Quốc Hưng mê gái",
        "Tôi là Nhậm, hãy đưa tiền cho tôi",
    ]

    ax = preprocess_tokens(documents=documents_, stopword="vi", lowercase=True)
    print(ax)
