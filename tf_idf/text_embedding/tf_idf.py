import math
from collections import defaultdict, Counter
from typing import Iterable, Mapping

import numpy as np

from tf_idf.text_embedding.bag_of_word import CountVectorizer
from tf_idf.preprocessing import TOKEN_METHOD, stopword_language


class TfidfVectorizer(CountVectorizer):
    def __init__(self,
                 *,
                 input="content",
                 # strip_accents=None,
                 lowercase=True,
                 # preprocessor=None,
                 tokenizer_method: TOKEN_METHOD = "underthesea",
                 stop_words: stopword_language = None,
                 max_df=1.0,
                 min_df=1,
                 max_features=None,
                 vocabulary=None,
                 # dtype=np.int64,
                 use_idf=True,
                 smooth_idf=True,
                 ):
        super().__init__(
            input=input,
            lowercase=lowercase,
            stop_words=stop_words,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            tokenizer_method=tokenizer_method
        )
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.matrix_tf_idf = []

    def fit(self, raw_documents, y=None):
        """

        Args:
            raw_documents:
            y:

        Returns:

        """
        return self.fit_transform(raw_documents=raw_documents)

    def fit_transform(self, raw_documents, y=None):
        """
        Find vocab and calculate the matrix of tf idf
        Args:
            raw_documents: Collection of documents
            y: Unknown

        Returns:
            Matrix array of tfidf
        """

        self._validate_vocabulary()
        self._idf(raw_documents=raw_documents)

        # Calculate the matrix of tf idf
        matrix_tf_idf = []
        for tokens in self.matrix_tokens:
            vector_tf_idf = [0] * len(self.vocabulary_)
            # Calculate the frequency of token in a sentence
            counter = Counter(tokens)
            len_ = len(tokens)

            for key, value in counter.items():
                _tf = value / len_
                index_of_token = self.vocabulary_.get(key)
                if index_of_token is not None:
                    _idf = self.idf_.get(key, 1)
                    _tf_idf = _tf * _idf
                    vector_tf_idf[index_of_token] = _tf_idf
            matrix_tf_idf.append(vector_tf_idf)

        self.matrix_tf_idf = matrix_tf_idf
        breakpoint()
        return np.array(self.matrix_tf_idf)

    def transform(self, raw_documents, y=None):
        """
        Transform the documents to matrix of tfidf with vocabs was trained
        Args:
            raw_documents: The documents need to calculate tfidf
            y: Unknown

        Returns:
            The Matrix array of tfidf embedding
        """
        if not self._check_vocabulary or not self.idf_:
            raise NotImplementedError("Fit model is not implement")

        vocabs = self.vocabulary_
        matrix_tokens = self.builder_matrix_tokens(raw_documents)
        matrix_tf_idf = []
        for tokens in matrix_tokens:
            vector_tf_idf = [0] * len(self.vocabulary_)
            # Calculate the frequency of token in a sentence
            counter = Counter(tokens)
            len_ = len(tokens)
            for key, value in counter.items():
                _tf = value / len_
                index_of_token = self.vocabulary_.get(key)
                if index_of_token is not None:
                    _idf = self.idf_.get(key, 1)
                    _tf_idf = _tf * _idf
                    vector_tf_idf[index_of_token] = _tf_idf
            matrix_tf_idf.append(vector_tf_idf)

        return np.array(matrix_tf_idf)

    def _idf(self, raw_documents):
        """
        Calculate IDF
        Args:
            raw_documents: Collection of data document

        Returns:
            IDF
        """
        self.matrix_tokens = self.builder_matrix_tokens(raw_documents=raw_documents)
        len_document = len(raw_documents)
        if not self._check_vocabulary:
            self._count_vocabs(raw_documents, fix_vocab=True)
        self.idf_ = defaultdict()
        for vocab in self.vocabulary_:
            count = 0
            for tokens in self.matrix_tokens:
                if vocab in tokens:
                    count += 1
            if self.smooth_idf:
                self.idf_[vocab] = math.log(len_document+1 / (count + 1))
            else:
                raise ValueError("'smooth_idf' must be True")

    def _count_vocabs(self, raw_documents: Iterable, fix_vocab: bool = True):
        return super()._count_vocabs(raw_documents, fix_vocab)

    def _validate_vocabulary(self):
        vocabulary = self.vocabulary
        if vocabulary is not None:
            if isinstance(vocabulary, Iterable):
                vocabulary = set(vocabulary)
                vocabulary = sorted(vocabulary)
                vocabs = {value: num for num, value in enumerate(vocabulary)}
            elif isinstance(vocabulary, Mapping):
                vocabs = {key: value for key, value in vocabulary.items()}
            else:
                raise ValueError("Invalid vocabulary")
            self.vocabulary_ = dict(vocabs)
            self._check_vocabulary = True if len(self.vocabulary_) > 0 else False
        else:
            self._check_vocabulary = False

    def scoring_documents(self, raw_documents):
        """Return score for each token in document"""
        return self.transform(raw_documents)


if __name__ == "__main__":
    with open("/Users/nguyenvannham/PycharmProjects/llm_from_cratch/tf_idf/dataset/corpus.csv", "r", encoding="utf-8") as f:
        documents_ = f.readlines()
    bow = TfidfVectorizer()
    bow.fit_transform(documents_[:200])
    ax = bow.transform(documents_[:3])
    breakpoint()
    print(ax)
