# from classification.classification_experiments import *
# import numpy as np
import pickle
from preprocessing.preprocess import clean_data_1, clean_data_2, stem
import pandas as pd
"""
model_analysis.py was used to conduct the misclassification analysis
and tweet analysis used in the project report. 

The tweets were posted by British musician Wiley before his account was
banned by Twitter as a consequence of them. 
More information regarding the Tweets can be found by using the link below.
https://www.bbc.co.uk/news/uk-53536471

@Author Siôn Davies
Date: August 2020
"""

# y_test, y_predict, model, X_values = make_multi_classification(tfidf_char, svc, True, False)
# misclassified = np.where(y_test != y_predict)
# predictions = np.asarray(y_predict)


# missclass = 'misclassified'
# with open(missclass, 'wb') as file:
#    pickle.dump(misclassified, file)

# pred = 'predictions'
# with open(pred, 'wb') as file:
#    pickle.dump(predictions, file)

# model, X, y = make_multi_classification(tfidf_char, svc, True, False)
# model.fit(X, y)
# filename = 'multi_model'
# with open(filename, 'wb') as file:
#    pickle.dump(model, file)

# model, X, y = make_classification(tfidf_char, svc, True)
# model.fit(X, y)
# filename ='binary_model'
# with open(filename, 'wb') as file:
#    pickle.dump(model, file)

# 1.) We check misclassified documents for the misclassification analysis.

with open('misclassified', 'rb') as file:
    miss_classed = pickle.load(file)

with open('predictions', 'rb') as file:
    predictions = pickle.load(file)


for sample in miss_classed:
    print(sample)

print('********************')
print(predictions[1469])

# 2.) Now for Twitter analysis, predicting on anti-semitic tweets.
print('********************')
with open('multi_model', 'rb') as file:
    multi_clf = pickle.load(file)

with open('binary_model', 'rb') as file:
    binary_clf = pickle.load(file)

df = pd.read_csv('../Datasets/dataset_utils/wiley_tweets.csv')
df['Message_Post'] = pd.DataFrame(df.Message_Post.apply(clean_data_1).apply(clean_data_2))
df.Message_Post = df.Message_Post.apply(stem)

tweet_1 = [df.Message_Post[0]]
tweet_2 = [df.Message_Post[1]]
tweet_3 = [df.Message_Post[2]]

print(multi_clf.predict(tweet_1))  # Predicted as 0 / 'Neither'
print(binary_clf.predict(tweet_1))  # Predicted as 0 / 'Non-fascist'
print(multi_clf.predict(tweet_2))  # Predicted as 0 / 'Neither'
print(binary_clf.predict(tweet_2))  # Predicted as 0 / 'Non-fascist'
print(multi_clf.predict(tweet_3))  # Predicted as 1 / 'Fascist'
print(binary_clf.predict(tweet_3))  # Predicted as 1 / 'Fascist'

