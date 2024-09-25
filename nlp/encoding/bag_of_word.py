import json
from typing import Dict, List, Tuple

import pandas as pd
from nlp.encoding.one_hot import OneHot


class BagOfWord(OneHot):
    """
    Implements a Bag-of-Words model for text encoding, extending from the OneHot class.
    """

    def fit(
        self,
        is_pyvi=True,
        vocab_cached_path: str = "nlp/cached/vocab_bag_of_word.json",
        use_cached=False,
        unknown_token="#sep",
    ) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Fits the vocabulary by tokenizing the input documents. Optionally, caches
        the vocabulary for future use.

        Args:
            is_pyvi (bool, optional): If True, uses the PyVi library for tokenization.
                Defaults to True.
            vocab_cached_path (str, optional): Path to save or load the cached vocabulary.
                Defaults to "nlp/cached/vocab_bag_of_word.json".
            use_cached (bool, optional): If True, loads the vocabulary from the cached file.
                If False, creates a new vocabulary from the input documents. Defaults to False.
            unknown_token (str, optional): Token used for unknown words. Defaults to "#sep".

        Returns:
            Tuple[Dict[str, int], Dict[int, str]]: A tuple containing the vocabulary
            (word-to-index mapping) and inverse vocabulary (index-to-word mapping).
        """
        if use_cached:
            with open(vocab_cached_path, "w", encoding="utf-8") as fp:
                vocab_map = json.load(fp)
                self.vocab, self.inverse_vocab = vocab_map.get("vocab"), vocab_map.get(
                    "inverse_vocab"
                )
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

    def transform(
        self, docs: List[str,], less_memory: True, is_pyvi=True
    ) -> Tuple | Dict | pd.DataFrame:
        """
        Transform documents to embedding vectors
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
        tokens = self.tokenizer_documents(documents=[sentence], is_pyvi=is_pyvi)
        embedd = {}

        for token in tokens[0]:
            index_token = vocab.get(token)

            if not index_token:
                index_token = 0

            if index_token in embedd:
                embedd[index_token] += 1
            else:
                embedd[index_token] = 1
        return embedd

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

