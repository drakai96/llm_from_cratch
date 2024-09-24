import json
from typing import Dict, List, Tuple

import pandas as pd
from nlp.encoding.one_hot import OneHot


class BagOfWord(OneHot):

    def fit(
        self,
        is_pyvi=True,
        vocab_cached_path: str = "nlp/cached/vocab_bag_of_word.json",
        use_cached=False,
        unknown_token="#sep",
    ) -> Tuple[Dict[str, int], Dict[int, str]]:

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

    def __transform_sentence(self, sentence: str, is_pyvi=True) -> Dict:
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
        data_encoder = pd.DataFrame(columns=[*self.vocab])

        for num, id_ in enumerate(ids):
            data_encoder.loc[num] = 0
            data_encoder.iloc[num, list(id_.keys())] = list(id_.values())

        return data_encoder

