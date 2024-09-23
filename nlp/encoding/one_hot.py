import json
from typing import List, Tuple, Dict


from base import BaseEncoder
from pydantic import Field
import pandas as pd


class OneHot(BaseEncoder):
    is_sklearn: bool = Field(
        default=False,
        description="Check is use sklearn library,\
     False if not use sklearn",
    )

    def fit(
        self,
        is_pyvi=True,
        vocab_cached_path: str = "nlp/cached/vocab_onehot.json",
        use_cached=False,
        unknown_token="#sep",
    ) -> Tuple[Dict[str, int], Dict[int, str]]:

        if use_cached:
            with open(vocab_cached_path, "w") as fp:
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
            import os

            print("root: ", os.getcwd())
            with open(vocab_cached_path, "w") as fp:
                json.dump(cache, fp=fp, indent=4, ensure_ascii=False)
        return self.vocab, self.inverse_vocab

    def __transform_sentence(self, sentence: str, is_pyvi=True) -> Tuple:
        """

        Args:
            sentence: Sentence need to embedding
            is_pyvi: bool = True
                Mean True if user pyvi library to tokenizer

        Returns:Tuple
            Vector tokenizer
        """
        vocab, _ = self.vocab, self.inverse_vocab
        tokens = self.tokenizer_documents(documents=[sentence], is_pyvi=is_pyvi)
        embedd = ()
        for token in tokens[0]:
            index_token = vocab.get(token)
            if index_token:
                embedd += (index_token,)
            else:
                embedd += (0,)
        return embedd

    def transform(
        self, docs: List[str,], less_memory: True, is_pyvi=True
    ) -> Tuple | Dict | pd.DataFrame:
        embedding = ()
        for doc in docs:
            embedding += (self.__transform_sentence(doc, is_pyvi=is_pyvi),)
        if not less_memory:
            embedding = self.__format_to_matrix(embedding)
        return embedding

    def vector_to_sentence(self, vector: List[int,]) -> str:
        _, inverse = self.vocab, self.inverse_vocab
        list_tokens = []
        for vec in vector:
            if inverse.get(vec):
                list_tokens.append(inverse.get(vec))
            else:
                list_tokens.append("#sep")
        sentence = " ".join(list_tokens)
        return sentence

    def __format_to_matrix(self, ids: Tuple) -> pd.DataFrame:
        data_encoder = pd.DataFrame(columns=[*self.vocab])
        for num, id_ in enumerate(ids):
            print(id_)
            data_encoder.loc[num] = 0
            data_encoder.iloc[num, list(set(id_))] = 1
            print(set(id_))
        return data_encoder


if __name__ == "__main__":
    docs = [
        "Trong thông báo gửi đi vào ngày 21/9, Bộ Tổng tham mưu Quân đội \
    nhân dân Việt Nam cho biết sẽ dừng huấn luyện diễu binh, diễu hành trong lễ kỷ niệm 80 năm \
    thành lập quân đội.",
        "Tôi muốn làm những người theo chủ nghĩa lý tưởng thất vọng, vì thực \
    tế là các bên ở châu Âu đều đang làm như vậy. Sự khác biệt giữa chúng tôi và những nước khác \
    nói chung là chúng tôi nói một cách trung thực và cởi mở về vấn đề này. Toàn bộ châu Âu đều làm \
    ăn với người Nga, nhưng một số nước phủ nhận điều này, chúng tôi không cần điều đó, ông Szijjarto \
    phát biểu tại Budapest vào hôm 20/9.",
    ]
    from nlp.preprocessing import CleanDocument

    clean = CleanDocument()
    docs = clean.clean_corpus(docs)

    onehot = OneHot(docs, is_sklearn=False)

    data = onehot.tokenizer_documents(is_pyvi=True)
    onehot_client = onehot.fit(
        use_cached=False,
        vocab_cached_path="/Users/nguyenvannham/PycharmProjects/pycharm_from_cratch/nlp/cached/vocab.json",
    )
    encode = onehot.transform(
        [
            "Việt Nam cho biết sẽ dừng huấn luyện diễu binh",
            "Trong thông báo gửi đi vào ngày",
            "Trong thông báo gửi đi vào ngày 21/9, Bộ Tổng tham mưu Quân đội \
    nhân dân Việt Nam cho biết sẽ dừng huấn luyện diễu binh, diễu hành trong lễ kỷ niệm 80 năm \
    thành lập quân đội.",
            "Khoảng 20.000 lệnh trừng phạt khác nhau đã được áp đặt đối với Nga. \
        Trên thực tế, chúng không thể được gọi là lệnh trừng phạt, vì lệnh trừng \
        phạt là biện pháp hợp pháp do Hội đồng Bảo an (Liên hợp quốc) đưa ra, trong \
        khi đây là các biện pháp cấm vận đơn phương",
        ],
        less_memory=False,
    )
    print("encode: ", encode)
    decode = onehot.vector_to_sentence([1, 6, 10000])
    print("decode: ", decode)
