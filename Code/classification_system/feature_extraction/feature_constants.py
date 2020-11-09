from feature_extraction.tfidf_features import *
from feature_extraction.word2vec_features import *
from feature_extraction.doc2vec_features import *
import gensim.downloader as api

"""
feature_constants.py contains constants to be used in the feature extraction
process and their respective getter methods. It also contains util methods
for the feature extraction process. 

@Author Siôn Davies
Date: July 2020
"""


def get_tfidf_word_vec():
    """
    Gets an instance of the Tf-idf word vectorizer.

    :return: Tf-idf word vectorizer.
    """
    tfidf_word = Tfidf_word_vec_features()
    return tfidf_word


def get_tfidf_char_vec():
    """
    Gets an instance of the Tf-idf char vectorizer.

    :return: Tf-idf char vectorizer.
    """
    tfidf_char = Tfidf_char_vec_features()
    return tfidf_char


def get_word2vec():
    """
    Gets an instance of the Word2Vec embedding vectorizer.

    :return: Word2Vec embedding vectorizer.
    """
    return Word2Vec_features(api.load("word2vec-google-news-300", return_path=True))


def get_doc2vec():
    """
    Gets a instance of the Doc2Vec embedding vectorizer.

    :return: Doc2Vec embedding vectorizer.
    """
    d2v = Doc2vec_features()
    return d2v


"""
Feature constants which are used as params in classification_experiments.py
"""
tfidf_word = get_tfidf_word_vec()
tfidf_char = get_tfidf_char_vec()
word2vec = get_word2vec()
doc2vec = get_doc2vec()
