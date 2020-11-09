from nltk.tokenize.treebank import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

"""
tfidf_features.py contains two Tf-idf vectorizer classes for text feature extraction.

Tfidf_word_vec_features is a transformer class that extracts word n-grams in the form of
uni-grams, bi-grams and tri-grams. 

Tfidf_char_vec features is a transformer class that extracts char n-grams. 

As both classes implement fit() and transform() methods they can be used as transformers
in the scikit-learn's  Pipeline. 

It utilizes scikit learns Tf-idfVectorizer which transforms text documents into a matrix
of Tf-idf features.

Author: Sion Davies
Date: July 2020 
"""


class Tfidf_word_vec_features:

    """
    Constructor: Loads in word vectors.
    analyzer = 'word' -> feature should be made up word n-grams.
    ngram_range = (1,3) -> word n-grams should be uni-grams, bi-grams and tri-grams.
    min_df = 2 -> when building vocab, ignore terms that appear too infrequently... in less than 2 documents.
    max_df = 0.25 -> remove terms that appear too frequently in the corpus... in 25%+ of documents.
    input -> default 'content' is input documents expected to be sequence of type String or byte .
    """

    def __init__(self):
        print("Loading vectors (...)")
        self.tfidf = TfidfVectorizer(
            strip_accents='unicode',
            tokenizer=TreebankWordTokenizer().tokenize,
            analyzer='word',
            input='content',
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.5,
            lowercase=True,
            smooth_idf=True,
        )

    def fit(self, X_data, y=None):
        return self.tfidf.fit(X_data)

    def transform(self, X_data, y=None):
        return self.tfidf.transform(X_data)

    def fit_transform(self, X_data, y=None):
        return self.tfidf.fit_transform(X_data)


class Tfidf_char_vec_features:

    """
    Constructor: Loads in char vectors.
    analyzer = 'char' -> features should be made of character n-grams.
    ngram_range = (2,6) -> char n_gram range.
    input -> default 'content' is input documents expected to be sequence of type String or byte .
    """

    def __init__(self):
        print("Loading vectors (...)")
        self.tfidf = TfidfVectorizer(
            strip_accents='unicode',
            tokenizer=TreebankWordTokenizer(),
            analyzer='char',
            input='content',
            ngram_range=(2, 6),
            lowercase=True,
            smooth_idf=True
        )

    def fit(self, X_data, y=None):
        return self.tfidf.fit(X_data)

    def transform(self, X_data, y=None):
        return self.tfidf.transform(X_data)

    def fit_transform(self, X_data, y=None):
        return self.tfidf.fit_transform(X_data)
