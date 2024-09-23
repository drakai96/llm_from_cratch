from abc import ABC, abstractmethod
from pyvi import ViTokenizer
from pydantic import Field
from typing import List, Tuple, Dict
import pandas as pd


class BaseEncoder(ABC):
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

        Args:
            is_pyvi: bool, default = True
                Mean True if used pyvi library to tokenizer
            vocab_cached_path: str default = nlp/cached/vocab.json
                Mean path to save cached vocab
            use_cached: bool, default = False
                Mean True if need to used cached to load vocab or save vocab
            unknown_token: str = #sep
                Mean token not in vocab change to #sep

        Returns:Dict[str, int], Dict[int, str]
            vocab and invert vocab

        """
        pass

    @abstractmethod
    def transform(self, docs: List[str,], less_memory: True) -> Tuple | pd.DataFrame:
        """

        Args:
            docs:
            less_memory:
        Returns:
            Tuple of vector embedding or DataFrame embedding
        """
        pass

    def __transform_sentence(self, sentence: str, is_pyvi=True):
        """

        Args:
            sentence: Sentence need to embedding
            is_pyvi: bool = True
                Mean True if user pyvi library to tokenizer

        Returns:Tuple
            Vector tokenizer
        """
        raise "Not implement"

    @abstractmethod
    def vector_to_sentence(self, vector: List[int,]) -> str:
        """

        Args:
            vector: List of idx in vocab

        Returns:
            convert idx to sentence
        """
        pass
