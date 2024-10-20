"""
This module used to embedding
"""
import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Union
import json


class OnehotEmbedding:
    """
    A class for creating one-hot embeddings from a corpus of text.
    """

    def __init__(self):
        """
        Initializes the OnehotEmbedding instance with vocab and invert_vocab.
        """
        self.vocab = {}
        self.invert_vocab = {}

    def fit(self, corpus: List[List[str]]) -> (dict, dict):
        """
        Builds the vocabulary from the given corpus.

        Args:
            corpus (List[List[str]]): A list of tokenized documents.

        Returns:
            Tuple[dict, dict]: The vocabulary and inverted vocabulary.
        """
        self.vocab = {}
        index = 1
        for tokens in corpus:
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = index
                    index += 1
        self.invert_vocab = {value: key for key, value in self.vocab.items()}
        return self.vocab, self.invert_vocab

    def transform_text_to_embed(self, tokens: List[str], reduce_memory: bool = True) -> List[int]:
        """
        Converts a list of tokens into one-hot or index-based embeddings.

        Args:
            tokens (List[str]): List of tokens to embed.
            reduce_memory (bool): If True, uses index representation; if False, one-hot encoding.

        Returns:
            List[int]: The one-hot or index-based embeddings.
        """
        if reduce_memory:
            embed_token = []
            for token in tokens:
                embed_token.append(self.vocab.get(token, 0))
            return embed_token
        embed_token = [0] * len(self.vocab)
        for token in tokens:
            encode_in_vocab = self.vocab.get(token)
            if encode_in_vocab:
                embed_token[encode_in_vocab] = 1
        return embed_token

    def invert_embed_to_text(self, embed_token: List[int], reduce_memory: bool = True) -> List[str]:
        """
        Converts embeddings back into tokens.

        Args:
            embed_token (List[int]): The embedding to convert.
            reduce_memory (bool): If True, assumes index representation; if False, one-hot encoding.

        Returns:
            List[str]: The corresponding tokens.
        """
        embed_token_copy = embed_token.copy()
        if not reduce_memory:
            embed_token_copy = []
            for idx, value in enumerate(embed_token):
                if value == 1:
                    embed_token_copy.append(idx)
        tokens = []
        for encode in embed_token_copy:
            tokens.append(self.invert_vocab.get(encode))
        return tokens


class TFIDF:
    """
    A class to compute TF-IDF values for a given set of documents.
    """

    def __init__(self, max_vocab=52_000, min_idf=0):
        """
        Initializes the TFIDF instance with a list of tokenized documents.
        """
        self.max_vocab = max_vocab
        self.min_idf = min_idf
        self.idf = {"unk":0}
        self.vocabs = {"unk":0}

    @staticmethod
    def _calculate_frequency(tokens: List[str]) -> Counter:
        """
        Returns: Frequency of token in sentence
        """

        return Counter(tokens)

    def __calculate_idf_and_vocab(self, corpus: List[List[str,]]) -> \
            Tuple[Dict[str, int], Dict[str, int]]:
        """
        Args:
            corpus:
        Returns:
            Idf cached, and vocab cached, it should save in disk
        """
        all_tokens = []
        for tokens in corpus:
            all_tokens.extend(tokens)
        counter = Counter(all_tokens)
        counter = {key: value for key, value in counter.items() if value > self.min_idf}

        # Calculate frequency of vocab
        counter = dict(sorted(counter.items(),
                              key=lambda item: item[1],
                              reverse=True)[:self.max_vocab])
        len_corpus = len(corpus)

        self.idf = {key: math.log(
            len_corpus / (value + 1))
            for key, value in counter.items()}

        # Map corpus to vocab index start from 1, equal zero if it not in corpus
        idx = 1
        for key in self.idf.keys():
            if key not in self.vocabs:
                self.vocabs[key] = idx
                idx += 1
        return self.idf, self.vocabs

    def fit(self, corpus: List[List[str]]):
        """
        Fit corpus
        Args:
            corpus: All documents in library

        Returns:
            idf cached of all corpus
        """

        return self.__calculate_idf_and_vocab(corpus=corpus)

    def fit_transform(self, tokens: List[str,]) -> List[float,]:
        """
        Transform a sentence to vector tfidf embedding
        Args:
            tokens: Token input
        """

        freq_tokens = self._calculate_frequency(tokens)

        vector_embed: list = [0] * (len(self.vocabs))
        len_sentence = len(tokens)
        for token in freq_tokens:
            id_vocab = self.vocabs.get(token, 0)
            tf = freq_tokens[token] / len_sentence
            idf = self.idf.get(token)
            vector_embed[id_vocab] = tf * idf
        return vector_embed

    def get_feature_name_out(self):
        """
        All token map in tfidf, it should use for columns with pandas vector
        """
        return list(self.vocabs.keys())

    @staticmethod
    def save_idf(idf, vocab, path: str = "tfidf_info.json") -> None:
        """

        Args:
            idf: idf cached dictionary was calculated by fit method
            vocab: vocab cached dictionary was calculated by fit method
            path: path need to save
        """
        tf_idf_info = {"idf": idf, "vocab": vocab}
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(tf_idf_info, fp=fp, indent=4, ensure_ascii=False)

    def load_tfidf_model(self, path_to_load: str = "tfidf_info.json") \
            -> Union[None, Tuple[dict, dict]]:
        """
        Load tfidf information, include idf and vocab of all corpus
        Args:
            path_to_load: path of tfidf vocab and if information
        Return:
            Load tfidf information, include idf and vocab of all corpus
        """
        if path_to_load.endswith(".json"):
            with open(path_to_load, "r", encoding="utf-8") as fp:

                info = json.load(fp)
                idf = info.get("idf")
                vocabs = info.get("vocabs")
                if idf and vocabs:
                    self.idf = idf
                    self.vocabs = vocabs
                    return self.idf, self.vocabs
                raise FileExistsError("File not has information of tfidf")
