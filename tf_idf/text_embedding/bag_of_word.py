from collections import Counter
from typing import Mapping, Iterable, List

import numpy as np
from tf_idf.preprocessing import preprocess_tokens, TOKEN_METHOD, stopword_language


class CountVectorizer:
    """
    Convert collection of text documents to matrix of frequency of each token
    Args:
        input: {"filename", "content"}, default = "content". Type of collection documents pass into class
            - If "filename" the collection documents are passed by read txt file

            - If "content" the collection documents are list of raw sentence

        # TODO strip_accents:
        lowercase: boolean, default = False
            - Convert all tokens to lower character after tokenizing
        # TODO preprocessor:

        tokenizer_method: ["pyvi","underthesea","whitespace"], default = "underthesea"
            - If "pyvi": Uses the PyVi library for Vietnamese tokenization.
            - If "underthesea": Uses the Underthesea library for Vietnamese tokenization.
            - If "split": Uses simple string splitting based on whitespace.

        stop_words: {"vi", "eng"}, default = None
            - If Vietnamese, built-in stop words for Vietnamese is used (load from datasets folder)
            - If None, stopword no used, but the tokens will depend on max df

        max_df: Float range(0,1), default = 1
            - Ignore tokens has document frequency higher than threshold

        min_df: int, default = 0
            - Ignore tokens has document frequency lower than threshold

        max_features: int, default = None
            - If None, all tokens used to build vocabs
            - Else, top tokens has higher frequency used to build vocabs
        vocabulary: Iterable of tokens. Example ["nham","van"]
            -
        dtype:
    """

    def __init__(self,
                 *,
                 input="content",
                 # strip_accents=None,
                 lowercase=True,
                 # preprocessor=None,
                 tokenizer_method: TOKEN_METHOD = "underthesea",
                 stop_words: stopword_language = None,
                 max_df=None,
                 min_df=None,
                 max_features=None,
                 vocabulary=None,
                 dtype=np.int64
                 ):
        """
        Init

        """
        self.input = input
        self.lowercase = lowercase
        self.tokenizer_method = tokenizer_method
        self.stop_words = stop_words
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.vocabulary = vocabulary
        self.dtype = dtype
        self.vocabulary_: dict = {}
        self.invert_vocabulary_: dict = {}

    def fit(self, raw_documents, y=None):
        """Transform raw documents to matrix frequency of vocabs"""
        # Calculate matrix token
        return self.fit_transform(raw_documents)

    def fit_transform(self, raw_documents, y=None):
        # Check exist and validate vocabulary
        self._validate_vocabulary()

        # calculate matrix tokens and creat vocabulary if it isn't exist
        _, x = self._count_vocabs(raw_documents=raw_documents, fix_vocab=self._check_vocabulary)

        x = self._limit_feature(matrix=x,
                                vocabulary=self.vocabulary_,
                                high=self.max_df,
                                low=self.min_df,
                                limit=self.max_features)
        return x

    def transform(self, raw_documents):

        _, x = self._count_vocabs(raw_documents=raw_documents, fix_vocab=False)
        return x

    def _validate_vocabulary(self):
        """
        Check and validate vocabulary to dictionary.
        If vocabulary exist we needn't fit documents
        Returns: None

        """
        vocabulary = self.vocabulary
        if vocabulary is not None:
            if isinstance(vocabulary, set):
                vocabulary = sorted(vocabulary)
            if not isinstance(vocabulary, Mapping):
                vocabs = {}
                for num, value in enumerate(vocabulary):
                    vocabs.setdefault(value, num)
                vocabulary = vocabs
            if len(vocabulary) == 0:
                raise ValueError("Empty vocabulary")
            self.vocabulary_ = vocabulary
            self._check_vocabulary = True if len(self.vocabulary_) > 0 else False
        else:
            self._check_vocabulary = False

    def _count_vocabs(self, raw_documents: Iterable, fix_vocab: bool = True):
        """
        Build matrix frequency of tokens and vocabulary dictionary
        Args:
            raw_documents:
            fix_vocab:

        Returns:

        """
        matrix_tokens = self.builder_matrix_tokens(raw_documents=raw_documents)
        if not fix_vocab:
            vocabulary = self.vocabulary_
        else:
            vocabs = set()
            vocabulary = {}
            for tokens in matrix_tokens:
                for token in tokens:
                    vocabs.add(token)
            vocabs = sorted(vocabs)

            for num, vocab in enumerate(vocabs):
                vocabulary.setdefault(vocab, num)

        self.vocabulary_ = vocabulary
        self._check_vocabulary = True
        self.invert_vocabulary_ = {value: key for key, value in self.vocabulary_.items()}

        # Build matrix tokens
        x = []
        for tokens in matrix_tokens:
            counter_token = Counter(tokens)
            counter_vec: list = [0] * len(self.vocabulary_)
            for key, value in counter_token.items():
                index_token_in_vocab = self.vocabulary_.get(key)
                if index_token_in_vocab:
                    counter_vec[int(index_token_in_vocab)] = value
            x.append(counter_vec)
        return vocabulary, x

    def builder_matrix_tokens(self, raw_documents: Iterable) -> List[List[str,]]:
        """
        Convert collection raw documents to matrix of tokens
        Args:
            raw_documents: Collection of document

        Returns:

        """
        matrix_tokens = preprocess_tokens(documents=raw_documents,
                                          method=self.tokenizer_method,
                                          lowercase=self.lowercase,
                                          stopword=self.stop_words)
        return matrix_tokens

    def _limit_feature(self, matrix, vocabulary, high=None, low=None, limit=None):
        """
        Beauty the matrix response
        Args:
            matrix:
            vocabulary:
            high:
            low:
            limit:

        Returns:

        """
        if not high and not low and not limit:
            return matrix, vocabulary

        matrix_tokens = np.array(matrix)
        vocab_frequency = np.sum(matrix_tokens, axis=0)
        prune_vocab = np.ones(len(vocabulary), dtype=bool)

        if high:
            prune_vocab &= vocab_frequency > high
        if low:
            prune_vocab &= vocab_frequency < low

        prune_index = np.where(prune_vocab is True)
        if limit:
            prune_index = prune_index[:limit]
        vocab = np.array(list(vocabulary.keys()))
        self.vocabulary_ = vocab[prune_index]
        x = matrix_tokens[:, prune_index]
        return x


if __name__ == "__main__":
    with open("./dataset/corpus.csv", "r", encoding="utf-8") as f:
        documents_ = f.readlines()
    bow = CountVectorizer()
    bow.fit_transform(documents_[:10])
    ax = bow.transform(documents_[:3])
    print(ax)
