from typing import Iterable

from bag_of_word import CountVectorizer
from tf_idf.preprocessing import TOKEN_METHOD, stopword_language


class TextOneHotEncoder(CountVectorizer):
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
                 ):
        """
        Init

        """
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
        self.vocabulary_ = {}

    def fit(self, raw_documents: Iterable[str], y=None):
        """

        Args:
            raw_documents: Collection of documents
            y:

        Returns:

        """
        return self.fit_transform(raw_documents=raw_documents, y=None)

    def fit_transform(self, raw_documents, y=None):
        """

        Args:
            raw_documents: Collection of documents
            y:

        Returns:

        """
        # Fit value
        x = super().fit_transform(raw_documents=raw_documents, y=None)

        x[x != 0] = 1
        return x

