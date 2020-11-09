import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.parsing.preprocessing import preprocess_string
from tqdm import tqdm
from sklearn import utils

"""
doc2vec_features.py contains the class Doc2vec_features.
Doc2vec_features is a Doc2Vec vectorizer class. It can be used as a feature extractor transformer in 
scikit-learn's Pipeline as it implements both fit() and transform() methods. 

Unlike the word vectors generated by Word2vec that generate a feature vector for every word in a corpus, 
Doc2vec extracts vector representations of variable length. Therefore, feature vectors are 
computed for each document in the corpus. 

Doc2Vec trains a neural network to derive paragraph vectors, it does this by training the network to predict 
the probability distribution of words within a paragraph. 

-

Code references: 

(C) was adapted to create the Paragraph vector Doc2vec class.

(C) Nag, A., (2019). A Text Classification Approach Using Vector Space Modelling(Doc2vec) & PCA. [online] Medium. 
Available at: 
<https://medium.com/swlh/a-text-classification-approach-using-vector-space-modelling-doc2vec-pca-74fb6fd73760> 
[Accessed 3 July 2020].

-
@Author: Siôn Davies
Date: July 2020
"""


class Doc2vec_features:

    """
    Constructor: Loads in document vectors.
    vector_size -> Dimensionality of feature vectors.
    window -> Max distance between the current and predicted word in a sentence.
    min_count -> Ignores all words with total frequency less than this number.
    epochs -> Number of times to iterate over corpus.
    dm -> if dm = 0: d2v model is PV-DBOW (distributed bag of words), if dm = 1: d2v model is PV-DM (distributed memory)
    workers -> How many worker threads to use to train model.
    """

    def __init__(self):
        # Doc2Vec constructor.
        print("Loading vectors (...)")
        self.model = None
        self.vector_size = 200
        self.window = 3
        self.min_count = 1
        self.epochs = 20
        self.dm = 0
        self.workers = 4
        print("Loading vectors completed.")

    def fit(self, X_data, y=None):
        # For each document in X_data, create a list of tokens using gensim's preprocess_string function/
        # Next, assign it a unique tag (i) to use as an input to train model using gensim's TaggedDocument().
        X_tagged_docs = [TaggedDocument(preprocess_string(document), [i]) for i, document in enumerate(X_data)]
        # Initialize model with constructor parameters.
        d2v_model = Doc2Vec(
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            dm=self.dm,
            workers=self.workers
        )
        # Build a vocabulary for the model using the tagged documents in X_data
        # Use tqdm to output a progress bar.
        d2v_model.build_vocab([x for x in tqdm(X_tagged_docs)])
        print('Doc2vec training commencing...')
        for epoch in range(self.epochs):
            print('\n')
            print('epoch: ' + str(epoch))
            # Train D2V model using the shuffled vocab in tagged X_data documents.
            # Repeat for given number of epochs.
            d2v_model.train(utils.shuffle([x for x in tqdm(X_tagged_docs)]), total_examples=len(X_tagged_docs),
                            epochs=1)
        print('\n' + 'Training finished.' + '\n')
        # d2v_model.save('/Users/siondavies/Desktop/NLP/Feature_Extraction/DOC2VEC/d2vmodel')
        self.model = d2v_model
        return self

    def transform(self, X_data, y=None):
        # infer_vector -> infer a vector for given post-training document in X_data, return in vector matrix.
        return np.asmatrix(np.array([self.model.infer_vector(preprocess_string(document))
                                     for i, document in enumerate(X_data)]))

    def fit_transform(self, X_data, y=None):
        self.fit(X_data)
        return self.transform(X_data)