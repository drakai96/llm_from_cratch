import numpy as np


class CountVectorizer:
    def __init__(
        self,
        *,
        input="content",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        stop_words=None,
        max_df=1.0,
        min_df=1.0,
        max_features=None,
        vocabulary=None,
        dtype=np.int64
    ):
        pass

    def fit(self, raw_documents, y=None):
        pass

    def fit_transform(self, raw_documents, y=None):
        pass

    def transform(self, raw_documents):
        pass
