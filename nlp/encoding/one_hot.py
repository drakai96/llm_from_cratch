"""
Module contain Onehot embedding class
"""

import json
from typing import Dict, List, Tuple
import os
from collections import defaultdict
import pandas as pd
from base import BaseEncoder
from pydantic import Field


class OneHot(BaseEncoder):
    is_sklearn: bool = Field(
        default=False,
        description="Check is use sklearn library,\
     False if not use sklearn",
    )

    def __init__(self, documents: List[str,], is_sklearn: False) -> None:
        super().__init__(documents, is_sklearn)
        self.vocab = {}
        self.inverse_vocab = {}

    def fit(
        self,
        documents: List[str,],
        is_pyvi=True,  # Check whether to use the Pyvi library to tokenize sentences
        unknown_token="#sep",
    ) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Args:
            is_pyvi: bool, default = True
                Mean True if used pyvi library to tokenizer
            unknown_token: str = #sep
                Mean token not in vocab change to #sep

        Returns:Dict[str, int], Dict[int, str]
            vocab and invert vocab

        """

        # Tokenize sentence to word
        token_documents = self.tokenize_documents(documents, is_pyvi=is_pyvi)

        # Create vocab dictionary
        self.vocab = defaultdict()
        for token_doc in token_documents:
            for num, token in enumerate(token_doc):
                self.vocab[token] = num
        self.vocab = dict(self.vocab)

        # Inverse vocab to inverse_vocab dictionary, which transform id to vocab
        self.inverse_vocab = {value: key for key, value in self.vocab.items()}

        return self.vocab, self.inverse_vocab

    def __transform_sentence(self, sentence: str, is_pyvi=True) -> Tuple:
        """
        Args:
            sentence: Sentence need to embedding
            is_pyvi: bool = True
                Mean True if user pyvi library to tokenizer

        Returns:Tuple
            Vector tokenizer
        """
        vocab, _ = self.vocab, self.inverse_vocab

        tokens = self.tokenize_documents(documents=[sentence], is_pyvi=is_pyvi)

        embedd = ()
        for token in tokens[0]:
            index_token = vocab.get(token)
            if index_token:
                embedd += (index_token,)
            else:
                embedd += (0,)

        return embedd

    def transform(
        self, docs: List[str,], less_memory: True, is_pyvi=True
    ) -> Tuple | Dict | pd.DataFrame:
        """
        Embedding the document
            Args:
                is_pyvi: Check whether to use the Pyvi library to tokenize sentences
                docs:
                less_memory:
            Returns:
                Tuple of vector embedding or DataFrame embedding
        """
        embedding = ()
        # Tokenize sentences
        for doc in docs:
            embedding += (self.__transform_sentence(doc, is_pyvi=is_pyvi),)
        # Transform token to embedding matrix
        if not less_memory:
            embedding = self.__format_to_matrix(embedding)

        return embedding

    def vector_to_sentence(self, vector: List[int,]) -> str:
        """
        Convert less memory vector to matrix embedding
        Args:
            vector: List of idx in vocab

        Returns:
            convert idx to sentence
        """
        _, inverse = self.vocab, self.inverse_vocab

        list_tokens = []
        for vec in vector:
            if inverse.get(vec):
                list_tokens.append(inverse.get(vec))
            else:
                list_tokens.append("#sep")

        sentence = " ".join(list_tokens)
        return sentence

    def __format_to_matrix(self, ids: Tuple) -> pd.DataFrame:
        data_encoder = pd.DataFrame(columns=[*self.vocab])

        for num, id_ in enumerate(ids):
            print(id_)
            data_encoder.loc[num] = 0
            data_encoder.iloc[num, list(set(id_))] = 1
            print(set(id_))

        return data_encoder
