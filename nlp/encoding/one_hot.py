"""
Module contain Onehot embedding class
"""
import json
from typing import Dict, List, Tuple

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
        super().__init__(documents,is_sklearn)
        self.vocab = {}
        self.inverse_vocab = {}

    def fit(
        self,
        is_pyvi=True,
        vocab_cached_path: str = "nlp/cached/vocab_onehot.json",
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
                Mean True if used cached to load vocab or save vocab
            unknown_token: str = #sep
                Mean token not in vocab change to #sep

        Returns:Dict[str, int], Dict[int, str]
            vocab and invert vocab

        """
        if use_cached:
            with open(vocab_cached_path, "w", encoding="utf-8") as fp:
                vocab_map = json.load(fp)
                self.vocab, self.inverse_vocab = vocab_map.get("vocab"), vocab_map.get(
                    "inverse_vocab"
                )
                fp.close()
            return self.vocab, self.inverse_vocab

        token_documents = self.tokenizer_documents(is_pyvi=is_pyvi)
        self.vocab = {unknown_token: 0}
        index = 1
        for token_doc in token_documents:
            for token in token_doc:
                if token not in self.vocab:
                    self.vocab[token] = index
                    index += 1

        self.inverse_vocab = {value: key for key, value in self.vocab.items()}

        if not use_cached and vocab_cached_path:
            cache = {"vocab": self.vocab, "inverse_vocab": self.inverse_vocab}
            with open(vocab_cached_path, "w", encoding="utf-8") as fp:
                json.dump(cache, fp=fp, indent=4, ensure_ascii=False)

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

        tokens = self.tokenizer_documents(documents=[sentence], is_pyvi=is_pyvi)

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
                docs:
                less_memory:
            Returns:
                Tuple of vector embedding or DataFrame embedding
        """
        embedding = ()
        for doc in docs:
            embedding += (self.__transform_sentence(doc, is_pyvi=is_pyvi),)
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

