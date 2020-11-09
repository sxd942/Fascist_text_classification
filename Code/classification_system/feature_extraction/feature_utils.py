import numpy as np
from nltk import sent_tokenize, word_tokenize

"""
Utils method for feature extraction process.

@Author Siôn Davies
Date: July 2020
"""


def w2v_tokenizer(X_data, y=None):
    """
    Tokenizer for Word2Vec embedding vectorizer.
    Utilizes nltk sent_tokenize and word_tokenize functions.

    :param X_data: input variable X.
    :param y: NONE
    :return: A vocab of documents from X data that have been tokenized.
    """
    tokenized_docs = []
    for doc in X_data:
        tokenized_sentences = []
        for sentence in sent_tokenize(doc):
            tokenized_sentences += word_tokenize(sentence)
        tokenized_docs.append(np.array(tokenized_sentences))
    vocab = np.array(tokenized_docs)
    return vocab
