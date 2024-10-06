from typing import Dict, List, Tuple

import pandas as pd
from nlp.encoding.one_hot import OneHot
from collections import defaultdict

class BagOfWord(OneHot):
    """
    Implements a Bag-of-Words model for text encoding, extending from the OneHot class.
    """

    def fit(
        self,
        documents: List[str,],
        is_pyvi=True,
        unknown_token="#sep",
    ) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Fits the vocabulary by tokenizing the input documents. Optionally, caches
        the vocabulary for future use.

        Args:
            documents: List of sentence
            is_pyvi (bool, optional): If True, uses the PyVi library for tokenization.
                Defaults to True.
            unknown_token (str, optional): Token used for unknown words. Defaults to "#sep".

        Returns:
            Tuple[Dict[str, int], Dict[int, str]]: A tuple containing the vocabulary
            (word-to-index mapping) and inverse vocabulary (index-to-word mapping).
        """
        # Paser list of sentence to matrix of token.
        # Each sentence parse to a vector
        token_documents = self.tokenize_documents(documents, is_pyvi=is_pyvi)

        self.vocab = defaultdict()
        for token_doc in token_documents:
            for num, token in enumerate(token_doc):
                if token not in self.vocab:
                    self.vocab[token] = num

        self.inverse_vocab = {value: key for key, value in self.vocab.items()}

        return self.vocab, self.inverse_vocab

    def transform(
        self, docs: List[str,], less_memory: True, is_pyvi=True
    ) -> Tuple | Dict | pd.DataFrame:
        """
        Transform documents to embedding vectors
        Args:
            is_pyvi:
            docs:
            less_memory:
        Returns:
            Tuple of vector embedding or DataFrame embedding
        """
        # Embedding
        embedding = ()
        for doc in docs:
            embedding += (self.__transform_sentence(doc, is_pyvi=is_pyvi),)

        if not less_memory:
            embedding = self.__format_to_matrix(embedding)
        return embedding

    def __transform_sentence(self, sentence: str, is_pyvi=True) -> Dict:
        """
        Transforms a single sentence into its Bag-of-Words vector representation.

        Args:
            sentence (str): The sentence to be transformed.
            is_pyvi (bool, optional): If True, uses PyVi for tokenization. Defaults to True.

        Returns:
            Dict: A dictionary where keys are token indices and values are token counts
            in the sentence.
        """

        vocab, _ = self.vocab, self.inverse_vocab
        tokens = self.tokenize_documents(documents=[sentence], is_pyvi=is_pyvi)
        embedd = defaultdict()

        for token in tokens[0]:
            index_token = vocab.get(token)
            # sep token
            if not index_token:
                index_token = 0
            if index_token in embedd:
                embedd[index_token] += 1
            else:
                embedd[index_token] = 1
        return dict(embedd)

    def __format_to_matrix(self, ids: Tuple) -> pd.DataFrame:
        """
        Converts the sparse vector representation into a dense matrix format.

        Args:
            ids (Tuple): Tuple containing the sparse vector embeddings for each document.

        Returns:
            pd.DataFrame: A DataFrame where each row represents a document, and columns
            represent token counts in the document.
        """

        data_encoder = pd.DataFrame(columns=[*self.vocab])

        for num, id_ in enumerate(ids):
            data_encoder.loc[num] = 0
            data_encoder.iloc[num, list(id_.keys())] = list(id_.values())

        return data_encoder
