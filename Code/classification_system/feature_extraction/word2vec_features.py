import numpy as np
from gensim.models import KeyedVectors
from feature_extraction.feature_utils import w2v_tokenizer
# nltk.download('punkt')

"""
word2vec_features.py contains the class Word2Vec_features.

Class Word2Vec_features is a word embedding vectorizer class. It can be used as a feature extractor 
transformer in scikit-learn's Pipeline as it implements both fit() and transform() methods. 

** Please be warned that the pre-trained vectors require roughly 4GB of memory to load if limit is not set...
Limit is currently set to 500000, which requires about 1/6 of this memory.**

The class utilizes the KeyedVectors function in the Gensim library, which maps entities (Strings) and
vectors. 
The Word2Vec_features class acts as a 'Mean Embedding Vectorizer' -> it takes the average of every word  
vector in a given document to generate a single feature vector for the entire document. 
-

Code references: 

(D) was adapted to create the mean word embedding Word2vec class (slide 18).

(E) contains reference to the pre-trained word vectors (Google) that are loaded in via the Gensim loader API. 

(D) Tyson, N., (2016). Word Embedding Models & Support Vector Machines For Text Classification. [online] Slideshare.net.
 Available at: 
 <https://www.slideshare.net/bokononisms/word-embedding-models-support-vector-machines-for-text-classification> 
 [Accessed 1 July 2020].

(E) code.google.com. (2013). Word2vec (Pre-Trained Word and Phrase Vectors). [online] Available at: 
<https://code.google.com/archive/p/word2vec/> [Accessed 7 July 2020].

-
@Author: Siôn Davies
Date: July 2020
"""


class Word2Vec_features:

    # Constructor: Loads in Google's pre-trained word vectors via gensims Keyedvectors.
    def __init__(self, path):
        print("Loading vectors (...)")
        # use -> path = api.load("word2vec-google-news-300", return_path = True)
        # Warning this requires 4GB+ of RAM... can be limited to 1/6 of this by...
        # ... uncommenting limit parameter.
        self.w2v = KeyedVectors.load_word2vec_format(
            path,
            binary=True,
            limit=500000,
        )
        self.dim = self.w2v.vector_size
        # d2v_model.save('/Users/siondavies/Desktop/NLP/Feature_Extraction/DOC2VEC/d2vmodel')

    def fit(self, X_data, y=None):
        return self

    # Tokenize the text, and generate feature vectors for each given document.
    def transform(self, X_data, tokenize=True, y=None):
        if tokenize is True:
            vocab = w2v_tokenizer(X_data)
            # If token in vocab (X_data) appears in pre-trained model find vectors for it.
            # If token in vocab doesn't appear in pre-trained model -> ignore it.
            # Return a single feature vector that is the mean of all word vectors in the given document
            return np.array([np.mean([self.w2v[w] for w in words if w in self.w2v]
                                     or [np.zeros(self.dim)], axis=0) for words in vocab])
    # If words in X_data have already been tokenized then.
        else:
            return np.array([np.mean([self.w2v[w] for w in words if w in self.w2v]
                                     or [np.zeros(self.dim)], axis=0) for words in X_data])

    def fit_transform(self, X_data, y=None):
        self.fit(X_data)
        return self.transform(X_data)

