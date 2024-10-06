"""
This module contains class for embedding documents.
It includes functionalities for keyword extraction and document submission.
"""

import math
from collections import Counter
from typing import List, Tuple

from pandas.core.api import DataFrame

from nlp.encoding.base import BaseEncoder


class TfIdf(BaseEncoder):
    """
    Embedding with TFIDF
    """

    def __init__(self, is_sklearn=False) -> None:
        """

        Args:
            is_sklearn: True if use sklearn

        """
        super().__init__(is_sklearn)
        self.vocab = {}
        self.inverse_vocab = {}
        self.idf = ()

    def fit(
        self,
        documents: List[str,],
        is_pyvi=True,
        unknown_token="#sep",
        smooth=True,
    ) -> Tuple:
        """
        Args:
            documents: List of sentence
            is_pyvi: bool, default = True
                Mean True if used pyvi library to tokenizer
            unknown_token: str = #sep
                Mean token not in vocab change to #sep
            smooth: bool(default = True)
                Mean tfidf = log((1+tf)*(1+idf))
            tfidf variant
        Returns:Dict[str, int], Dict[int, str]
            vocab and idf cached vocab
        """

        # Paser string to token
        documents_tokenize = self.tokenize_documents(documents, is_pyvi=is_pyvi)

        # Create vocab
        self.vocab = {unknown_token: 0}
        index = 1
        for token_doc in documents_tokenize:
            for token in token_doc:
                if token not in self.vocab:
                    self.vocab[token] = index
                    index += 1

        # tfidf variant
        if smooth:
            d = len(documents) + 1
            for word in self.vocab.keys():
                count = 0
                for doc in documents:
                    if word in doc:
                        count += 1
                self.idf += (math.log(d + 1 / (count + 1)),)
        else:
            raise ValueError("Smooth = False is not implement")

        return self.vocab, self.idf

    def transform(self, docs: List[str], less_memory: True) -> Tuple | DataFrame:
        """

        Args:
            docs:
            less_memory:
        Returns:
            Tuple of vector embedding or DataFrame embedding
        """

        # If Less_memory = True memory will optimize
        if less_memory:
            embedding = ()
            for num, doc in enumerate(docs):
                emb, index_ = self.__transform_sentence(doc, num)
                tuple_embed_map = ((num, index_[i], emb[i]) for i in range(len(index_)))
                embedding += tuple(tuple_embed_map)
            return embedding

        # If Less_memory = False dataframe of embedding will return
        word_as_columns = list(self.vocab.keys())
        data_encoder = DataFrame(columns=word_as_columns)
        for num, doc in enumerate(docs):
            emb, index_ = self.__transform_sentence(doc, num)
            data_encoder.loc[num, :] = 0
            data_encoder.iloc[num, index_] = emb
        return data_encoder

    def __transform_sentence(self, sentence: str, is_pyvi=True):
        """
        Embedding text to t
        Args:
            sentence: Sentence need to embedding
            is_pyvi: bool = True
                Mean True if user pyvi library to tokenizer

        Returns:Tuple
            Vector tokenizer
        """
        tokens_sentence = self.tokenize_documents([sentence])[0]

        # Mapping unknown token to '#sep'
        tokens_sentence = [
            token if token in self.vocab else "#sep" for token in tokens_sentence
        ]

        # Counting frequency of list token
        token_count_dict = Counter(tokens_sentence)
        len_tokens_sentences = len(tokens_sentence)

        tf_idf = tuple()
        index_word = tuple()

        # Calculate tfidf and get index word
        for token in tokens_sentence:
            idx = self.vocab.get(token)

            if idx is None:
                idx = 0
            value = token_count_dict.get(token) / len_tokens_sentences
            tf_idf += ((value / len_tokens_sentences) * self.idf[idx],)
            index_word += (idx,)
        return tf_idf, index_word

    def vector_to_sentence(self, vector: List[int,]) -> str:
        raise AttributeError("Module not implement")


if __name__ == "__main__":
    documents_ = [
        (
            "Bình luận trên được đưa ra sau khi Tổng thống Ukraiane Volodymyr"
            "Zelensky cho rằng cuộc xung đột giữa Ukraine và Nga đang tiến gần"
            "đến hồi kết hơn"
            "nhiều người nghĩ. Ông cũng kêu gọi các đồng minh tăng cường năng "
            "lực cho Ukraine"
        ),
        (
            "Những ngày gần đây, ông Zelensky và giới chức Ukraine đề cập đến kế "
            "hoạch chiến thắng trong cuộc xung đột với Nga. Ông Zelensky sẽ trao đổi"
            "kế hoạch này với Tổng thống Mỹ Joe Biden và 2 ứng viên tổng thống nhân "
            "chuyến thăm Mỹ trong tuần này."
        ),
    ]
    from nlp.preprocessing.text_processing import TextPreprocessing

    clean_client = TextPreprocessing()
    import sys

    sys.path.append("/Users/nguyenvannham/PycharmProjects/llm_from_cratch/nlp")
    tfidf = TfIdf()
    tfidf.fit(documents_)
    ax = tfidf.transform(
        [(
                "hoạch chiến thắng trong cuộc xung đột với Nga." 
                "Ông Zelensky sẽ trao đổi")],
        less_memory=True,
    )
    ax2 = tfidf.transform(
        [
            (
             "hoạch chiến thắng trong cuộc xung đột"
             "với Nga. Ông Zelensky sẽ trao đổi"
            ),
            (
             "cuộc xung đột giữa Ukraine"
             " và Nga đang tiến gần đến hồi"),
        ],
        less_memory=True,
    )
    print(ax2)
    breakpoint()
    ax2.to_csv("cached/encoder.csv", index=False)
