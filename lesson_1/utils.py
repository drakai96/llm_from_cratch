from typing import Dict, List
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from embedding import TFIDF
from preprocessing import clean_text, tokenize_text

# Load corpus and model
with open("corpus/corpus.csv", "r", encoding="utf-8") as fp:
    CORPUS = fp.readlines()

MODEL = TFIDF()
if os.path.exists("vocab.json"):
    MODEL.load_tfidf_model("vocab.json")

DB = pd.read_csv("corpus/db.csv")


def pd_document_to_db(corpus: List[str]) -> List[List[str]]:
    """Transforms a list of sentences into a tokenized vocabulary.

    Args:
        corpus: A list of sentences.

    Returns:
        A list of tokenized sentences.
    """
    tokens_vocab = []
    for sentence in tqdm(corpus):
        tokens = document_to_token(sentence)
        tokens_vocab.append(tokens)
    return tokens_vocab


def make_db(n: int = 100) -> bool:
    """Creates a TF-IDF database from the corpus.

    Args:
        n: The number of documents to process from the corpus.

    Returns:
        True if the database is successfully created.
    """
    corpus_df = pd.read_csv("corpus/corpus.csv")
    corpus_list = corpus_df["corpus"].to_list()[:n]
    tokenized_corpus = pd_document_to_db(corpus_list)

    tf_idf = TFIDF()
    tf_idf.fit(tokenized_corpus)

    data = []
    for tokens in tokenized_corpus:
        transformed_tokens = tf_idf.fit_transform(tokens)
        data.append(transformed_tokens)

    # Save the processed data
    data_df = pd.DataFrame(data, columns=tf_idf.get_feature_name_out())
    data_df.to_csv("corpus/db.csv", index=False)
    tf_idf.save_idf(idf=tf_idf.idf, vocab=tf_idf.vocabs)

    return True


def find_importance_word(sentence: str, number_word_return: int = 10) -> Dict[str, float]:
    """Finds the most important words in a sentence using TF-IDF.

    Args:
        sentence: The input sentence.
        number_word_return: The number of top important words to return.

    Returns:
        A dictionary of important words and their scores.
    """
    query_tokens = document_to_token(sentence)
    embed_vector = MODEL.fit_transform(query_tokens)
    columns = list(MODEL.vocabs.keys())

    data_df = pd.DataFrame([embed_vector], columns=columns)
    top_words = data_df.iloc[0, :].nlargest(number_word_return)

    return top_words.to_dict()


def get_important_word(idx: int, data: pd.DataFrame, number_word_return: int = 10) -> Dict[str, float]:
    """Gets the most important words from a document in the database.

    Args:
        idx: The index of the document.
        data: The database as a pandas DataFrame.
        number_word_return: The number of top important words to return.

    Returns:
        A dictionary of important words and their scores.

    Raises:
        ValueError: If the number of words to return is greater than 1000.
    """
    if number_word_return > 1000:
        raise ValueError("Invalid number of words to return: must be <= 1000.")

    top_words = data.iloc[idx, :].nlargest(number_word_return)
    return top_words.to_dict()


def document_to_token(sentence: str) -> List[str]:
    """Cleans and tokenizes a sentence.

    Args:
        sentence: The input sentence.

    Returns:
        A list of tokens.
    """
    cleaned_sentence = clean_text(sentence, lower=False)
    tokens = tokenize_text(cleaned_sentence, token_method="underthesea")

    return tokens


def search(sentence: str, top_k: int = 10) -> List[str]:
    """Searches for the most relevant documents based on cosine similarity.

    Args:
        sentence: The query sentence.
        top_k: The number of top documents to return.

    Returns:
        A list of top-k most relevant documents from the corpus.
    """
    # Copy the database and convert it to a numpy array
    data_copy = DB.copy().to_numpy()

    # Tokenize and embed the query sentence
    query_tokens = document_to_token(sentence)
    query_vector = MODEL.fit_transform(query_tokens).reshape(1, -1)

    # Compute cosine similarity
    document_similarity = cosine_similarity(query_vector, data_copy)

    # Find indices of top-k relevant documents
    top_indices = np.argpartition(document_similarity, -top_k)[0][:top_k]
    top_documents = [CORPUS[i] for i in top_indices]

    return top_documents
