"""
This module contains class for embedding documents.
It includes functionalities for keyword extraction and document submission.
"""
import json
import math
from collections import Counter
from typing import List, Tuple

from pandas.core.api import DataFrame

from nlp.encoding.base import BaseEncoder


class TfIdf(BaseEncoder):
    """
    Embedding with TFIDF
    """

    def __init__(self, documents: List[str], is_sklearn=False) -> None:
        """

        Args:
            documents:
            is_sklearn:

        """
        super().__init__(documents, is_sklearn)
        self.vocab = {}
        self.inverse_vocab = {}
        self.idf = ()

    def fit(
            self,
            is_pyvi=True,
            vocab_cached_path: str = "nlp/cached/vocab_tfidf.json",
            use_cached=False,
            unknown_token="#sep",
            smooth=True,
    ) -> Tuple:
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
            smooth: bool(default = True)
                Mean tfidf = log((1+tf)*(1+idf))
            tfidf variant
        Returns:Dict[str, int], Dict[int, str]
            vocab and idf cached vocab
        """
        if use_cached:
            with open(vocab_cached_path, "w", encoding="utf-8") as fp:
                vocab_map = json.load(fp)
                self.vocab, self.inverse_vocab = vocab_map.get("vocab"), vocab_map.get(
                    "idf_cached"
                )
            return self.vocab, self.inverse_vocab

        # Paser string to token
        token_documents = self.tokenizer_documents(is_pyvi=is_pyvi)

        # Create vocab
        self.vocab = {unknown_token: 0}
        index = 1
        for token_doc in token_documents:
            for token in token_doc:
                if token not in self.vocab:
                    self.vocab[token] = index
                    index += 1

        # tfidf variant
        if smooth:
            d = len(self.documents) + 1
            for word in self.vocab.keys():
                count = 0
                for doc in self.documents:
                    if word in doc:
                        count += 1
                self.idf += (math.log(d + 1 / (count + 1)),)
        else:
            raise ValueError("Smooth = False is not implement")

        # Cached vocab and idf with vocab_cached_path
        if not use_cached and vocab_cached_path:
            cache = {"vocab": self.vocab, "idf_cached": self.idf}
            with open(vocab_cached_path, "w", encoding="utf-8") as fp:
                json.dump(cache, fp=fp, indent=4, ensure_ascii=False)
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
        tokens_sentence = self.tokenizer_documents([sentence])[0]

        # Mapping unknown token to '#sep'
        tokens_sentence = [token if token in self.vocab else "#sep" for token in tokens_sentence]

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
    documents_ = [("Bình luận trên được đưa ra sau khi Tổng thống Ukraiane Volodymyr"
                   "Zelensky cho rằng cuộc xung đột giữa Ukraine và Nga đang tiến gần"
                   "đến hồi kết hơn"
                   "nhiều người nghĩ. Ông cũng kêu gọi các đồng minh tăng cường năng "
                   "lực cho Ukraine"),
                  ("Những ngày gần đây, ông Zelensky và giới chức Ukraine đề cập đến kế "
                   "hoạch chiến thắng trong cuộc xung đột với Nga. Ông Zelensky sẽ trao đổi"
                   "kế hoạch này với Tổng thống Mỹ Joe Biden và 2 ứng viên tổng thống nhân "
                   "chuyến thăm Mỹ trong tuần này.")]
    from nlp.preprocessing.text_processing import CleanDocument

    clean_client = CleanDocument()
    documents_ = clean_client.clean_corpus(documents_)
    import sys

    sys.path.append("/Users/nguyenvannham/PycharmProjects/llm_from_cratch/nlp")
    tfidf = TfIdf(documents_)
    tfidf.fit(
        vocab_cached_path="cached/vocab_tfidf.json")
    ax = tfidf.transform([
        ("hoạch chiến thắng trong cuộc xung đột với Nga."
         "Ông Zelensky sẽ trao đổi")
    ],
        less_memory=True)
    print(ax)

    ax2 = tfidf.transform(
        [
            ("hoạch chiến thắng trong cuộc xung đột"
             "với Nga. Ông Zelensky sẽ trao đổi"),
            ("cuộc xung đột giữa Ukraine"
             " và Nga đang tiến gần đến hồi")],
        less_memory=False)
    print(ax2)
    ax2.to_csv("cached/encoder.csv", index=False)
