import re
import string
import nltk
import inflect
from nltk.corpus import stopwords

"""
preprocess.py contains the methods used to pre-process the data. 
The textual data is normalized and cleaned, with non-sensical and 
irrelevant data being discarded. 

Functions: token_lemma and lemmatize were decided not to be used
in the pre-process phase. Stemming was opted for instead. 

Functions clean_data 1 & 2 are applied to the data before they were
split and saved into training and test sets in the first pre-processing
phase. This process can be observed in the dataset creation files in the 
repository.
 
The second pre-processing phase occurs after the data has been gathered
in classification_experiments.py, where the text is stemmed and has English stop-
words removed. 

@Author: Sion Davies
Date: July 2020
"""


def remove_emoticons(data):
    """
    Function to remove emoticons from text.

    :param data: The text data to be searched.
    :return: The text data with emoticons removed.
    """
    emoticons = regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  
        u"\U0001F300-\U0001F5FF"  
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F1E0-\U0001F1FF" "]+", flags = re.UNICODE)
    return emoticons.sub(r'', data)


def convert_numbers(data):
    """
    Function to replace numerical numbers with their text counterparts.

    :param data: The text data to be searched.
    :return: The text data with numerical numbers replaced with textual representation.
    """
    inf = inflect.engine()
    for word in data:
        if word.isdigit():
            data = re.sub(word, inf.number_to_words(word), data)
        else:
            continue
    return data


def token_lemma(data):
    """
    Function to simultaneously tokenize and lemmatize text.

    :param data: The text data to be searched.
    :return: The text data which has been both tokenized and lemmatized.
    """
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokenizer.tokenize(data)]


def lemmatize(data):
    """
    Function to lemmatize text.

    :param data: The text data to be searched.
    :return: The text data which has been lemmatized.
    """
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in data]


def stem(data):
    """
    Function to stem text and remove stop-words.

    :param data: The text documents in the corpus.
    :return: The text documents with english stop-words removed,
    whose words have been stemmed.
    """
    stemmer = nltk.stem.SnowballStemmer('english')
    stop_words = stopwords.words('english')
    words = data.split()
    stemmed_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(stemmed_words)


def remove_stopwords(data):
    """
    Function to remove stop-words.

    :param data: The text documents in the corpus.
    :return: The text documents with english stop-words removed.
    """
    stop_words = stopwords.words('english')
    words = data.split()
    new_words = [word for word in words if word not in stop_words]
    return ' '.join(new_words)


def clean_data_1(data):
    """
    First step of the pre-processing phase.
    Converts String to lower case.
    Deletes text between < and >
    Removes punctuation from text.
    Removes URLs

    :param data: Text data to be cleaned.
    :return: Cleaned text data.
    """
    data = data.lower()
    data = re.sub('<.*?>', '', data)
    data = re.sub('[%s]' % re.escape(string.punctuation), '', data)
    data = re.sub(r'http\S+', '', data)
    return data


def clean_data_2(data):
    """
    Second step of pre-processing phase.
    Removes non-sensical data.
    Removes emoticons.
    Converts numerical numbers to textual equivalent.
    Clears up white space.

    :param data: Text data to be cleaned.
    :return: Cleaned text data.
    """
    data = re.sub('-', ' ', data)
    data = re.sub('\n', '', data)
    data = remove_emoticons(data)
    data = convert_numbers(data)
    data = re.sub(' +', ' ', data)
    return data

