"""
Base class for encoding
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import pandas as pd
from pydantic import Field
from pyvi import ViTokenizer


class BaseEncoder(ABC):
    """
       Abstract base class for text encoders, which can tokenize and transform
       text documents into vector embeddings.

       Attributes:
           __AtriPyvi__ (bool): Indicates whether the PyVi library is used
               for tokenizing sentences.
    """
    __AtriPyvi__: bool = Field(
        default=True,
        description="Check is use pyvi library\
     to token sentences",
    )

    def __init__(self, documents: List[str,], is_sklearn: False) -> None:
        self.documents = documents
        self.is_sklearn = is_sklearn

    def tokenizer_documents(self, documents: List[str] = None, is_pyvi: bool = True):
        """
        Tokenizes the input documents using PyVi or a simple split method.
        Args:
            documents: Document of sentence default = False
            is_pyvi: bool, default = True
                Mean if True, the documents will be token with pyvi library,\
                 if False, the documents will be token by split-text method

        Returns:
            List token documents

        """
        if not documents:
            documents = self.documents

        if is_pyvi:
            token_docs = []
            for doc in documents:
                tokens = ViTokenizer.tokenize(doc).split()
                _token_sentence = []
                for token in tokens:
                    _token_sentence.append(token.replace("_", " "))
                token_docs.append(_token_sentence)
        else:
            token_docs = [doc.split() for doc in documents]

        return token_docs

    @abstractmethod
    def fit(
        self,
        is_pyvi=True,
        vocab_cached_path: str = "nlp/cached/vocab.json",
        use_cached=False,
        unknown_token="#sep",
    ) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Fit documents to vocab corpus and invert vocab corpus
        Args:
            is_pyvi: bool, default = True
                Mean True if used pyvi library to tokenizer
            vocab_cached_path: str default = nlp/cached/vocab.json
                Mean path to save cached vocab
            use_cached: bool, default = False
                Mean True if used cached to load vocab or save vocab
            unknown_token: str = #sep
                Mean token not in vocab change to #sep

        Returns:
            Tuple[Dict[str, int], Dict[int, str]]: A tuple containing the vocabulary
            (word-to-index mapping) and its inverse (index-to-word mapping).

        """

    @abstractmethod
    def transform(self, docs: List[str,], less_memory: True) -> Tuple | pd.DataFrame:
        """
        Transforms the input documents into vector embeddings.
        Args:
            docs:
            less_memory:
        Returns:
            Tuple of vector embedding or DataFrame embedding
        """

    def __transform_sentence(self, sentence: str, is_pyvi=True):
        """
        Transforms a single sentence into a vector representation.
        Args:
            sentence: Sentence need to embedding
            is_pyvi: bool = True
                Mean True if user pyvi library to tokenizer

        Returns:Tuple
            Vector tokenizer
        """

    @abstractmethod
    def vector_to_sentence(self, vector: List[int,]) -> str:
        """

        Args:
            vector: List of idx in vocab

        Returns:
            convert idx to sentence
        """
