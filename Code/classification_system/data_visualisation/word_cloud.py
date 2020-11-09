import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from wordcloud import WordCloud
"""
word_cloud.py was used to generate Word-Cloud of fascist and hate speech
samples. 

@Author: Siôn Davies
Date: July 2020
"""
# First we create the fascist word cloud.

df = pd.read_csv('../Datasets/dataset_utils/Gold_cleaned.csv')


def converter(Fascist_Speech):
    if Fascist_Speech == 'Yes':
        return 1
    else:
        return 0


df['Cluster'] = df['Fascist_Speech'].apply(converter)
fascist = df[df.Cluster == 1]
fascist_samples = fascist['Message_Post'].str.split(' ')


def get_words(data):
    combined_fascist_samples = []
    for sample in data:
        sample = [word for word in sample]
        combined_fascist_samples.append(sample)
    return combined_fascist_samples


combined_fascist_samples = get_words(fascist_samples)


def generate_text(data):
    fascist_text = [' '.join(text) for text in data]
    generate_fascist_samples = ' '.join(fascist_text)
    return generate_fascist_samples


words = generate_text(combined_fascist_samples)

stop_words = stopwords.words('english')
# Adding irrelevant words to stop words list.
stop_words.append('think'), stop_words.append('thing'), stop_words.append('im')
stop_words.append('will'), stop_words.append('one'), stop_words.append('people')
stop_words.append('would'), stop_words.append('like'), stop_words.append('even')
stop_words.append('time'), stop_words.append('still'), stop_words.append('hi')
stop_words.append('way'), stop_words.append('really'), stop_words.append('dont')
stop_words.append('know'), stop_words.append('idea'), stop_words.append('also')
stop_words.append('good'), stop_words.append('though'), stop_words.append('see')
stop_words.append('us'), stop_words.append('get')

fascist_wc = WordCloud(stopwords=stop_words, background_color="white").generate(words)

plt.figure(figsize=(10, 10))
plt.imshow(fascist_wc, interpolation='bilinear')
plt.axis("off")
plt.show()

# Now creating the hate word cloud...

df_hate = pd.read_csv('../Datasets/Multiclass/Hate_Fascist_Gold.csv')
df_hate = df_hate[df_hate.Numeric_Label == 2]
hate_samples = df_hate['Message_Post'].str.split(' ')
combined_hate_samples = get_words(hate_samples)
hate_words = generate_text(combined_hate_samples)

hate_wc = WordCloud(stopwords=stop_words, background_color="white").generate(hate_words)

plt.figure(figsize=(10, 10))
plt.imshow(hate_wc, interpolation='bilinear')
plt.axis("off")
plt.show()

