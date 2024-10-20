from typing import List, Dict

import pandas as pd
from tqdm import tqdm

from embedding import TFIDF
from preprocessing import clean_text, tokenize_text


def pd_document_to_db(corpus: List[str,]):
    """
    Transform list sentence to tfidf vector db
    Args:
        corpus: List of sentence

    Returns:

    """
    tokens_vocab = []
    for sentence in tqdm(corpus):
        tokens = document_to_token(sentence=sentence)
        tokens_vocab.append(tokens)

    return tokens_vocab


def make_db(n=100):
    corpus_ = pd.read_csv("/Users/nguyenvannham/PycharmProjects/llm_from_cratch/nlp/lesson1/corpus/corpus.csv")
    corpus_1 = corpus_["corpus"].to_list()[:n]
    corpus_1 = pd_document_to_db(corpus_1)
    tf_idf = TFIDF()
    tf_idf.fit(corpus_1)
    data = []
    for tokens in corpus_1:
        print(tokens)
        ax = tf_idf.fit_transform(tokens=tokens)
        data.append(ax)
    data_pd = pd.DataFrame(data, columns=tf_idf.get_feature_name_out())
    data_pd.to_csv("/Users/nguyenvannham/PycharmProjects/llm_from_cratch/nlp/lesson1/corpus/db.csv", index=False)
    return True


def get_important_word(idx: int, data: pd.DataFrame, number_word_return: int = 10) -> Dict[str, float]:
    if number_word_return > 1000:
        raise ValueError("invalid n")
    value = data.iloc[idx, :].nlargest(number_word_return)
    return value.to_dict()


def document_to_token(sentence: str) -> List[str,]:
    sentence = clean_text(sentence)
    tokens = tokenize_text(sentence)
    return tokens


if __name__ == "__main__":
    # corpus_ = pd.read_csv("/Users/nguyenvannham/PycharmProjects/llm_from_cratch/nlp/lesson1/corpus/corpus.csv")
    # corpus_1 = corpus_["corpus"].to_list()[:30]
    # corpus_1 = pd_document_to_db(corpus_1)
    # tf_idf = TFIDF()
    # ax = tf_idf.fit(corpus_1)
    # data = []
    # for tokens in corpus_1:
    #     print(tokens)
    #     ax = tf_idf.fit_transform(tokens=tokens)
    #     data.append(ax)
    # print(corpus_1[0])
    # make_db()
    data = pd.read_csv("/Users/nguyenvannham/PycharmProjects/llm_from_cratch/nlp/lesson1/corpus/db.csv")
    ax = get_important_word(2, data)
    print(ax)
