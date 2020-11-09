import pandas as pd
import matplotlib.pyplot as plt
from nltk import ngrams
from preprocessing.preprocess import remove_stopwords
"""
frequent_ngrams.py was used to generate bar plots of most frequently used
bi and trigrams from the fascist and hate documents. 

@Author: Siôn Davies
Date: July 2020
"""
# First the fascist documents...
df = pd.read_csv('../Datasets/dataset_utils/Gold_cleaned.csv')
df.Message_Post = df.Message_Post.apply(remove_stopwords)


def converter(Fascist_Speech):
    if Fascist_Speech == 'Yes':
        return 1
    else:
        return 0


df['Numeric_Label'] = df['Fascist_Speech'].apply(converter)
fascist = df[df.Numeric_Label == 1]


def list_format(data):
    words = data.split()
    return [word for word in words]


words = list_format(''.join(str(fascist.Message_Post.tolist())))
bigrams_series = (pd.Series(ngrams(words, 2)).value_counts())[:12]
trigrams_series = (pd.Series(ngrams(words, 3)).value_counts())[:12]

bigrams_series.sort_values().plot.barh(color='navy', width=0.7, figsize=(7, 3))
plt.ylabel('Bigram')
plt.xlabel('Frequency')
plt.show()

trigrams_series.sort_values().plot.barh(color='navy', width =0.7, figsize=(7, 4))
plt.ylabel('Trigram')
plt.xlabel('Frequency')
plt.show()

# Now to do the same for the hate documents...

df_hate = pd.read_csv('../Datasets/Multiclass/Hate_Fascist_Gold.csv')
df_hate.Message_Post = df_hate.Message_Post.apply(remove_stopwords)
hate = df_hate[df_hate.Label == 2]

hate_words = list_format(''.join(str(hate.Message_Post.tolist())))
hate_bigrams_series = (pd.Series(ngrams(hate_words, 2)).value_counts())[:12]
hate_trigrams_series = (pd.Series(ngrams(hate_words, 3)).value_counts())[:12]

hate_bigrams_series.sort_values().plot.barh(color='navy', width=0.7, figsize=(7, 3))
plt.ylabel('Bigram')
plt.xlabel('Frequency')
plt.show()

hate_trigrams_series.sort_values().plot.barh(color='navy', width =0.7, figsize=(7, 4))
plt.ylabel('Trigram')
plt.xlabel('Frequency')
plt.show()
