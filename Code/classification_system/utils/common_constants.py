import nltk
from sklearn.model_selection import StratifiedKFold
"""
common_constants.py contains constants to be used in classification process.

@Author Siôn Davies
Date: July 2020
"""

stop_words = nltk.word_tokenize(' '.join(nltk.corpus.stopwords.words('english')))
GOLD = 'Gold: '
SHUFFLED = 'Shuffled: '
SR = 'SR: '
SYNTHETIC = 'Synthetic: '

CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

