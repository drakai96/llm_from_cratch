"""
This module used to embedding
"""
from typing import List, Dict
from collections import defaultdict
import math


class OnehotEmbedding:
    """
    A class for creating one-hot embeddings from a corpus of text.
    """

    def __init__(self):
        """
        Initializes the OnehotEmbedding instance with vocab and invert_vocab.
        """
        self.vocab = {}
        self.invert_vocab = {}

    def fit(self, corpus: List[List[str]]) -> (dict, dict):
        """
        Builds the vocabulary from the given corpus.

        Args:
            corpus (List[List[str]]): A list of tokenized documents.

        Returns:
            Tuple[dict, dict]: The vocabulary and inverted vocabulary.
        """
        self.vocab = {}
        index = 1
        for tokens in corpus:
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = index
                    index += 1
        self.invert_vocab = {value: key for key, value in self.vocab.items()}
        return self.vocab, self.invert_vocab

    def transform_text_to_embed(self, tokens: List[str], reduce_memory: bool = True) -> List[int]:
        """
        Converts a list of tokens into one-hot or index-based embeddings.

        Args:
            tokens (List[str]): List of tokens to embed.
            reduce_memory (bool): If True, uses index representation; if False, one-hot encoding.

        Returns:
            List[int]: The one-hot or index-based embeddings.
        """
        if reduce_memory:
            embed_token = []
            for token in tokens:
                embed_token.append(self.vocab.get(token, 0))
            return embed_token
        else:
            embed_token = [0] * len(self.vocab)
            for token in tokens:
                encode_in_vocab = self.vocab.get(token)
                if encode_in_vocab:
                    embed_token[encode_in_vocab] = 1
            return embed_token

    def invert_embed_to_text(self, embed_token: List[int], reduce_memory: bool = True) -> List[str]:
        """
        Converts embeddings back into tokens.

        Args:
            embed_token (List[int]): The embedding to convert.
            reduce_memory (bool): If True, assumes index representation; if False, one-hot encoding.

        Returns:
            List[str]: The corresponding tokens.
        """
        embed_token_copy = embed_token.copy()
        if not reduce_memory:
            embed_token_copy = []
            for idx, value in enumerate(embed_token):
                if value == 1:
                    embed_token_copy.append(idx)
        tokens = []
        for encode in embed_token_copy:
            tokens.append(self.invert_vocab.get(encode))
        return tokens


class TFIDF:
    """
    A class to compute TF-IDF values for a given set of documents.
    """

    def __init__(self, documents: List[List[str]]):
        """
        Initializes the TFIDF instance with a list of tokenized documents.

        Args:
            documents (List[List[str]]): A list of tokenized documents.
        """
        self.documents: List[List[str]] = documents
        self.tf: List[Dict[str, float]] = []
        self.idf: Dict[str, float] = {}
        self.tfidf: List[Dict[str, float]] = []

        self._calculate_tf()
        self._calculate_idf()
        self._calculate_tfidf()

    def _calculate_tf(self):
        """Calculate term frequency for each document."""
        for doc in self.documents:
            words = doc
            term_count = defaultdict(int)
            for word in words:
                term_count[word] += 1
            # Normalize term frequency
            total_terms = len(words)
            self.tf.append({word: count / total_terms for word, count in term_count.items()})

    def _calculate_idf(self):
        """Calculate inverse document frequency for each term."""
        total_documents = len(self.documents)
        df = defaultdict(int)  # Document frequency for each term

        for tf_dict in self.tf:
            for term in tf_dict.keys():
                df[term] += 1

        for term, count in df.items():
            self.idf[term] = math.log(total_documents / count)

    def _calculate_tfidf(self):
        """Calculate TF-IDF for each document."""
        for tf_dict in self.tf:
            tfidf_dict = {term: tf * self.idf[term] for term, tf in tf_dict.items()}
            self.tfidf.append(tfidf_dict)

    def get_tfidf(self) -> List[Dict[str, float]]:
        """Return the TF-IDF values for all documents."""
        return self.tfidf


# Example usage
if __name__ == "__main__":
    documents_ = [
        ["Nguyen", "Van", "Nham"],
        ["Nham", "Van", "Nguyen"],
        ["Nguyen", "Quoc", "Hung", "me", "gai"],
    ]

    tfidf = TFIDF(documents_)
    result = tfidf.get_tfidf()

    for idx_, doc_tfidf in enumerate(result):
        print(f"Document {idx_ + 1}: {doc_tfidf}")

    documents_ = [
        ["Nguyen", "Van", "Nham"],
        ["Nham", "Van", "Nguyen"],
        ["Nguyen", "Quoc", "Hung", "me", "gai"],
    ]

    # Initialize OnehotEmbedding
    onehot = OnehotEmbedding()

    # Fit the model to the corpus
    vocab, invert_vocab = onehot.fit(documents_)
    print("Vocabulary:", vocab)
    print("Inverted Vocabulary:", invert_vocab)

    # Transform text to embeddings
    tokens_ = ["the", "cat", "sat"]
    embeddings = onehot.transform_text_to_embed(tokens_, reduce_memory=True)
    print("Embeddings for tokens", tokens_, ":", embeddings)

    # Invert embeddings back to text
    inverted_tokens = onehot.invert_embed_to_text(embeddings, reduce_memory=True)
    print("Inverted tokens from embeddings:", inverted_tokens)
