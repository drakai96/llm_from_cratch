from bag_of_word import CountVectorizer


class TfidfVectorizer(CountVectorizer):
    pass

    def scoring_documents(self, raw_documents):
        """Return score for each token in document"""
