import os

from tf_idf.text_embedding.tf_idf import TfidfVectorizer

# Get the absolute path to the project directory
base_dir = os.path.dirname(os.path.abspath(__file__))
corpus_path = os.path.join(base_dir, "../dataset/corpus.csv")
with open("../dataset/corpus.csv", "r") as fb:
    raw_document = fb.readlines()[:10]
tf_idf = TfidfVectorizer()
tf_idf.fit(raw_documents=raw_document, )


class KeywordExtraction:

    def __init__(self, word_extract_engine=tf_idf):
        self.word_extract_engine = word_extract_engine

    def extract(self, document):
        tfidf_value = self.word_extract_engine.transform([document])
        tf_idf_ = {list(self.word_extract_engine.vocabulary_.keys())[i]: tfidf_value[0][i] for i in
                   range(len(self.word_extract_engine.vocabulary_))}
        sorted_dict = dict(sorted(tf_idf_.items(), key=lambda item: item[1], reverse=True))

        return sorted_dict


if __name__ == "__main__":
    keyword = KeywordExtraction()
    with open("../dataset/test.txt", "r", encoding="utf-8") as fp:
        document_ = fp.read()

    ax_ = keyword.extract(document_)
    print(ax_)