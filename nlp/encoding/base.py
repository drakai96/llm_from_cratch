"""
Base class for encoding
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Literal
import json
import pickle
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

    def __init__(self, is_sklearn: False) -> None:
        """
        Init class
        Args:
            is_sklearn:
        """
        self.is_sklearn = is_sklearn

    @staticmethod
    def tokenize_documents(documents: List[str], is_pyvi: bool = True):
        """
        Tokenize the input documents using PyVi or a simple split method.
        Args:
            documents: Document of sentence default = False
            is_pyvi: bool, default = True
                Mean if True, the documents will be token with pyvi library,\
                 if False, the documents will be token by split-text method

        Returns:
            List token documents

        """

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
        documents: List[str,],
        is_pyvi=True,
        unknown_token="#sep",
    ) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Fit documents to vocab corpus and invert vocab corpus
        Args:
            documents: List of sentence need to embedding
            is_pyvi: bool, default = True
                Mean True if used pyvi library to tokenizer
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

    @staticmethod
    def save_embedding(path: str, vocab_embedding: Any) -> None:
        """

        Args:
            path: path to save embedding model ('.json' or '.pickle')
            vocab_embedding: the embedding model
        Returns:

        """
        # Save model embedding as json file
        if path.endswith(".json"):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(f, vocab_embedding)

        # Save model embedding as pickle file
        if path.endswith(".json"):
            with open(path, "w+", encoding="utf-8") as f:
                pickle.dump(f, vocab_embedding)

    @staticmethod
    def load_embedding(path: str, path_type: Literal["json", "pickle"] = None):
        """

        Args:
            path: The path of embedding model
            path_type: The format of file (json or pickle)

        Returns:
            Any: The loaded embedding data.
        """
        if os.path.exists(path):
            raise FileNotFoundError(f"The path file '{path}' do not exits.")

        # Checking format of path
        if not path_type:
            if path.endswith(".pickle"):
                path_type = "pickle"
            elif path.endswith(".json"):
                path_type = ".json"
            else:
                raise ValueError("Please specify path file '.json' or 'pickle' ")

        # Load pickle embedding model
        if path_type == ".pickle":
            with open(path, "rb", encoding="utf-8") as f:
                embedding_model = pickle.load(f)
                return embedding_model

        # Load json embedding model
        if path_type == ".json":
            with open(path, "r", encoding="utf-8") as f:
                embedding_model = json.load(f)
                return embedding_model
